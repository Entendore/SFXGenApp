#!/usr/bin/env python3

"""
SFX Generator Pro v4 - Single-file PySide6 sound-effects synthesizer.
40 SFX types, 15 FX, FFT spectrum, spectrogram, layering, presets,
history, keyboard shortcuts, A/B compare, waveform zoom, VU meter,
undo/redo, favorites, search, export formats, playback animation.
"""

import numpy as np
import soundfile as sf
from scipy import signal as sig
from scipy.ndimage import uniform_filter1d
import io, os, wave, json, time, sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QSlider, QPushButton, QGroupBox, QGridLayout,
    QFileDialog, QMessageBox, QScrollArea, QSizePolicy, QStatusBar,
    QSplitter, QTabWidget, QListWidget, QListWidgetItem, QCheckBox,
    QSpinBox, QDoubleSpinBox, QInputDialog, QMenu, QToolBar,
    QToolButton, QLineEdit, QWidgetAction, QProgressDialog,
)
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QPoint, QRect
from PySide6.QtGui import (
    QPainter, QPen, QColor, QLinearGradient, QFont,
    QPalette, QPainterPath, QAction, QKeySequence, QShortcut, QImage,
)

SAMPLE_RATE = 44100
DEFAULT_SEED = None  # None = random each time


# -- Helpers -----------------------------------------------------------------

def _set_seed(seed):
    if seed is not None:
        np.random.seed(int(seed))


def _to_mono_float(samples):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    peak = np.max(np.abs(samples))
    if peak > 1e-6:
        samples = samples / peak
    return samples.astype(np.float32)


def _to_stereo(samples, pan=0.0):
    """Convert mono to stereo with pan -1..+1."""
    left_gain = np.cos((pan + 1) * np.pi / 4)
    right_gain = np.sin((pan + 1) * np.pi / 4)
    stereo = np.column_stack([samples * left_gain, samples * right_gain]).astype(np.float32)
    return stereo


def _envelope_adsr(n, a=0.01, d=0.05, s=0.6, r=0.1, sr=SAMPLE_RATE):
    a_n = int(min(a, n))
    d_n = int(min(d, n - a_n))
    r_n = int(min(r, n - a_n - d_n))
    sus = max(n - a_n - d_n - r_n, 0)
    env = np.ones(n, dtype=np.float64)
    if a_n > 0:
        env[:a_n] = np.linspace(0, 1, a_n, endpoint=False)
    if d_n > 0:
        env[a_n:a_n + d_n] = np.linspace(1, s, d_n, endpoint=False)
    if sus > 0:
        env[a_n + d_n:a_n + d_n + sus] = s
    if r_n > 0:
        env[-r_n:] = np.linspace(s, 0, r_n, endpoint=False)
    return env


def _noise(n, rng=None):
    return rng.uniform(-1, 1, n) if rng is not None else np.random.uniform(-1, 1, n)


def _apply_fade(samples, fade_in_ms=0, fade_out_ms=0, sr=SAMPLE_RATE):
    n = len(samples)
    out = samples.astype(np.float64)
    fi = int(fade_in_ms / 1000 * sr)
    fo = int(fade_out_ms / 1000 * sr)
    if fi > 0 and fi < n:
        out[:fi] *= np.linspace(0, 1, fi, endpoint=False)
    if fo > 0 and fo < n:
        out[-fo:] *= np.linspace(1, 0, fo, endpoint=False)
    return out.astype(np.float32)


# -- Post-processing FX ------------------------------------------------------

def apply_reverb(samples, room_size=0.3, damping=0.5, sr=SAMPLE_RATE):
    """Reverb via parallel comb filters - optimized with fewer larger delays and batch lfilter."""
    n = len(samples)
    x = samples.astype(np.float64)
    # Use 4 delays instead of 8 - halve filter count, similar sound
    delays_ms = [29.1, 41.3, 61.7, 89.3]
    feedback = room_size * (0.85 - damping * 0.25)
    out = np.zeros(n, dtype=np.float64)
    for d_ms in delays_ms:
        d = int(d_ms * sr / 1000)
        if d >= n:
            continue
        b = np.zeros(d + 1)
        b[0] = 1.0
        a = np.zeros(d + 1)
        a[0] = 1.0
        a[d] = -feedback
        out += sig.lfilter(b, a, x) * 0.2
    out += x  # Add dry signal
    peak = np.max(np.abs(out))
    if peak > 1e-6:
        out /= peak
    return out.astype(np.float32)


def apply_delay(samples, delay_ms=150, feedback=0.35, mix=0.3, sr=SAMPLE_RATE):
    """Echo / delay effect (vectorized)."""
    n = len(samples)
    d = int(delay_ms * sr / 1000)
    if d <= 0 or d >= n:
        return samples
    out = np.copy(samples).astype(np.float64)
    wet = np.zeros(n)
    current = np.copy(samples)
    for i in range(12):
        offset = d * (i + 1)
        if offset >= n:
            break
        end = min(offset + len(current), n)
        length = end - offset
        wet[offset:end] += current[:length] * (feedback ** i)
    out = out * (1 - mix) + wet * mix
    peak = np.max(np.abs(out))
    if peak > 1e-6:
        out /= peak
    return out.astype(np.float32)


def apply_distortion(samples, drive=2.0, sr=SAMPLE_RATE):
    """Soft-clip distortion via tanh waveshaper."""
    out = np.tanh(samples.astype(np.float64) * drive)
    peak = np.max(np.abs(out))
    if peak > 1e-6:
        out /= peak
    return out.astype(np.float32)


def apply_lowpass(samples, cutoff_hz=3000, sr=SAMPLE_RATE):
    """2nd-order Butterworth low-pass filter (vectorized)."""
    nyq = sr / 2
    cutoff = min(cutoff_hz, nyq - 1)
    b, a = sig.butter(2, cutoff / nyq, btype='low')
    out = sig.lfilter(b, a, samples.astype(np.float64))
    peak = np.max(np.abs(out))
    if peak > 1e-6:
        out /= peak
    return out.astype(np.float32)


def apply_highpass(samples, cutoff_hz=100, sr=SAMPLE_RATE):
    """2nd-order Butterworth high-pass filter (vectorized)."""
    nyq = sr / 2
    cutoff = min(cutoff_hz, nyq - 1)
    b, a = sig.butter(2, cutoff / nyq, btype='high')
    out = sig.lfilter(b, a, samples.astype(np.float64))
    peak = np.max(np.abs(out))
    if peak > 1e-6:
        out /= peak
    return out.astype(np.float32)


def apply_pitch_shift(samples, semitones=0, sr=SAMPLE_RATE):
    """Resample-based pitch shift."""
    if semitones == 0:
        return samples
    factor = 2.0 ** (semitones / 12.0)
    orig_len = len(samples)
    indices = np.arange(0, orig_len, factor)
    indices = indices[indices < orig_len].astype(int)
    pitched = samples[indices]
    target = np.linspace(0, len(pitched) - 1, orig_len)
    out = np.interp(target, np.arange(len(pitched)), pitched)
    peak = np.max(np.abs(out))
    if peak > 1e-6:
        out /= peak
    return out.astype(np.float32)


def apply_reverse(samples):
    return samples[::-1].astype(np.float32)


def apply_chorus(samples, rate_hz=1.5, depth_ms=7, mix=0.3, sr=SAMPLE_RATE):
    """Chorus effect via modulated delay (vectorized)."""
    n = len(samples)
    mod_depth = int(depth_ms * sr / 1000)
    if mod_depth < 2:
        return samples
    t = np.arange(n, dtype=np.float64) / sr
    mod = np.sin(2 * np.pi * rate_hz * t) * mod_depth
    delay_samples = (mod + mod_depth).astype(int)
    delay_samples = np.clip(delay_samples, 0, n - 1)
    # Build delayed signal using fancy indexing
    indices = np.arange(n) - delay_samples
    indices = np.clip(indices, 0, n - 1)
    wet = samples[indices].astype(np.float64)
    out = samples.astype(np.float64) * (1 - mix) + wet * mix
    peak = np.max(np.abs(out))
    if peak > 1e-6:
        out /= peak
    return out.astype(np.float32)


def apply_compressor(samples, threshold_db=-12, ratio=4.0, sr=SAMPLE_RATE):
    """Simple dynamic range compressor."""
    x = samples.astype(np.float64)
    abs_x = np.abs(x) + 1e-10
    db = 20 * np.log10(abs_x)
    gain = np.where(db > threshold_db,
                    threshold_db + (db - threshold_db) / ratio - db,
                    0.0)
    gain = np.minimum(gain, 0)  # Only compress, don't expand
    gain_linear = 10 ** (gain / 20)
    out = x * gain_linear
    peak = np.max(np.abs(out))
    if peak > 1e-6:
        out /= peak
    return out.astype(np.float32)


def apply_tremolo(samples, rate_hz=5, depth=0.5, sr=SAMPLE_RATE):
    """Amplitude tremolo effect."""
    n = len(samples)
    t = np.arange(n, dtype=np.float64) / sr
    lfo = 1.0 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t))
    out = samples.astype(np.float64) * lfo
    peak = np.max(np.abs(out))
    if peak > 1e-6:
        out /= peak
    return out.astype(np.float32)


def apply_eq(samples, low_gain_db=0, mid_gain_db=0, high_gain_db=0, sr=SAMPLE_RATE):
    """3-band parametric EQ (low/mid/high shelving)."""
    x = samples.astype(np.float64)
    nyq = sr / 2
    # Low shelf at 300 Hz
    if low_gain_db != 0:
        b, a = sig.butter(1, 300 / nyq, btype='low')
        low = sig.lfilter(b, a, x)
        gain = 10 ** (low_gain_db / 20)
        x = x + low * (gain - 1)
    # High shelf at 3000 Hz
    if high_gain_db != 0:
        b, a = sig.butter(1, 3000 / nyq, btype='high')
        high = sig.lfilter(b, a, x)
        gain = 10 ** (high_gain_db / 20)
        x = x + high * (gain - 1)
    # Mid bell at 1000 Hz
    if mid_gain_db != 0:
        w0 = 1000.0 / sr
        if 0 < w0 < 1:
            b, a = sig.iirpeak(w0, 1.0, 10 ** (mid_gain_db / 40))
            x = sig.lfilter(b, a, x)
    peak = np.max(np.abs(x))
    if peak > 1e-6:
        x /= peak
    return x.astype(np.float32)


def apply_flanger(samples, rate_hz=0.5, depth_ms=3, feedback=0.5, mix=0.5, sr=SAMPLE_RATE):
    """Flanger effect - vectorized via chunked fractional delay with numpy."""
    n = len(samples); x = samples.astype(np.float64)
    fb = min(feedback, 0.9)
    max_delay_samples = max(int(depth_ms * sr / 1000) + 2, 4)
    t = np.arange(n, dtype=np.float64) / sr
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t)
    delay_line = lfo * (max_delay_samples - 2) + 1  # varying delay per sample
    # Process in chunks for vectorization
    chunk_size = 2048
    # Use a circular buffer sized for the full signal
    buf_len = n + max_delay_samples + 2
    buf = np.zeros(buf_len)
    out = np.zeros(n)
    for c_start in range(0, n, chunk_size):
        c_end = min(c_start + chunk_size, n)
        c_len = c_end - c_start
        # Write input into buffer at absolute position (no wrap needed)
        buf[c_start:c_end] = x[c_start:c_end]
        # Compute delay indices
        delays = delay_line[c_start:c_end]
        int_delays = delays.astype(int)
        frac_delays = delays - int_delays
        int_delays = np.clip(int_delays, 1, max_delay_samples)
        # Read delayed samples from earlier in buffer
        read_starts = c_start - int_delays
        read_starts = np.clip(read_starts, 0, n - 1)
        # Linear interpolation between adjacent samples
        idx0 = read_starts
        idx1 = np.clip(read_starts + 1, 0, n - 1)
        delayed = buf[idx0] * (1 - frac_delays) + buf[idx1] * frac_delays
        # Mix dry/wet
        out[c_start:c_end] = x[c_start:c_end] * (1 - mix) + delayed * mix
        # Write feedback for future reads
        feedback_sig = np.clip(x[c_start:c_end] + delayed * fb, -10, 10)
        buf[c_start:c_end] = feedback_sig
    peak = np.max(np.abs(out))
    if peak > 1e-6: out /= peak
    return out.astype(np.float32)


def apply_phaser(samples, rate_hz=0.5, depth=0.5, feedback=0.3, mix=0.5, sr=SAMPLE_RATE):
    """Phaser effect - vectorized via chunked scipy.lfilter for each allpass stage."""
    n = len(samples); x = samples.astype(np.float64)
    # Base allpass center frequencies (Hz)
    center_freqs = [400, 800, 1200, 2400, 3600, 4800]
    t = np.arange(n, dtype=np.float64) / sr
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t)  # 0..1
    nyq = sr / 2
    # Use ~44 chunks (one per ~10ms)
    chunk_size = max(int(0.01 * sr), 256)
    phase_sig = x.copy()
    for k, cf in enumerate(center_freqs):
        freq_sweep = cf * (0.3 + 0.7 * lfo * (1 - k * 0.1))
        processed = np.zeros(n)
        for c_start in range(0, n, chunk_size):
            c_end = min(c_start + chunk_size, n)
            avg_f = float(np.mean(freq_sweep[c_start:c_end]))
            avg_f = np.clip(avg_f, 20, nyq - 1)
            w0 = avg_f / sr
            if w0 <= 0 or w0 >= 0.5:
                continue
            q = 0.707 + depth * 4.0
            b, a = sig.iirnotch(w0, q)
            processed[c_start:c_end] = sig.lfilter(b, a, phase_sig[c_start:c_end])[0]
        phase_sig = phase_sig * (1 - mix * 0.3) + processed * mix * 0.3
    out = x * (1 - mix) + phase_sig * mix
    peak = np.max(np.abs(out))
    if peak > 1e-6: out /= peak
    return out.astype(np.float32)


def apply_bitcrusher(samples, bit_depth=8, sample_rate_reduce=1.0, sr=SAMPLE_RATE):
    """Bitcrusher / lo-fi effect."""
    x = samples.astype(np.float64); n = len(x)
    if bit_depth < 1: bit_depth = 1
    if bit_depth >= 16: return samples
    levels = 2 ** int(bit_depth)
    crushed = np.round(x * levels) / levels
    if sample_rate_reduce < 1.0 and sample_rate_reduce > 0.01:
        step = max(1, int(1.0 / sample_rate_reduce))
        for i in range(0, n - step, step):
            crushed[i+1:i+step] = crushed[i]
    peak = np.max(np.abs(crushed))
    if peak > 1e-6: crushed /= peak
    return crushed.astype(np.float32)


def apply_stereo_width(samples, width=1.0, sr=SAMPLE_RATE):
    """Adjust stereo width. 0=mono, 1=normal, >1=wide. Returns stereo."""
    if samples.ndim == 1:
        stereo = np.column_stack([samples, samples])
    else:
        stereo = samples.copy()
    mid = (stereo[:, 0] + stereo[:, 1]) * 0.5
    side = (stereo[:, 0] - stereo[:, 1]) * 0.5
    stereo[:, 0] = mid + side * width
    stereo[:, 1] = mid - side * width
    peak = np.max(np.abs(stereo))
    if peak > 1e-6: stereo /= peak
    return stereo.astype(np.float32)


def normalize_audio(samples, mode="peak", target_db=-3.0, sr=SAMPLE_RATE):
    """Normalize audio. Modes: peak, rms, lufs."""
    x = samples.astype(np.float64)
    if mode == "peak":
        peak = np.max(np.abs(x))
        if peak > 1e-6:
            target = 10 ** (target_db / 20.0)
            x = x * (target / peak)
    elif mode == "rms":
        rms = np.sqrt(np.mean(x ** 2))
        if rms > 1e-6:
            target = 10 ** (target_db / 20.0)
            x = x * (target / rms)
    elif mode == "lufs":
        # Simplified LUFS approximation using K-weighting
        if x.ndim == 1:
            x_stereo = np.column_stack([x, x])
        else:
            x_stereo = x.copy()
        # K-weighting: high shelf +4dB at ~1.5kHz, high-pass ~38Hz
        nyq = sr / 2
        # High shelf (simplified): boost high freqs
        b_shelf, a_shelf = sig.butter(1, 1500 / nyq, btype='low')
        low_part = sig.lfilter(b_shelf, a_shelf, x_stereo, axis=0)
        high_part = x_stereo - low_part
        k_weighted = x_stereo + high_part * 1.0  # ~4dB shelf boost
        # High-pass at 38Hz
        b_hp, a_hp = sig.butter(2, 38 / nyq, btype='high')
        k_weighted = sig.lfilter(b_hp, a_hp, k_weighted, axis=0)
        # Mean square with 400ms gating blocks
        block_size = int(0.4 * sr)
        if len(k_weighted) > block_size:
            n_blocks = len(k_weighted) // block_size
            ms_blocks = np.zeros(n_blocks)
            for b in range(n_blocks):
                block = k_weighted[b * block_size:(b + 1) * block_size]
                if block.ndim == 2:
                    ms_blocks[b] = np.mean(block[:, 0] ** 2 + block[:, 1] ** 2) * 0.5
                else:
                    ms_blocks[b] = np.mean(block ** 2)
            # Remove blocks below -70 LUFS (absolute gate)
            gate_thresh = 10 ** (-70 / 10.0)
            gated = ms_blocks[ms_blocks > gate_thresh]
            if len(gated) > 0 and np.mean(gated) > 0:
                lufs = -0.691 + 10 * np.log10(np.mean(gated))
            else:
                lufs = -70.0
        else:
            lufs = -70.0
        # Apply gain to reach target LUFS
        gain_db = target_db - lufs
        gain = 10 ** (gain_db / 20.0)
        x = x * gain
    else:
        return samples
    peak = np.max(np.abs(x))
    if peak > 1.0: x = x / peak  # Prevent clipping
    return x.astype(np.float32)


def apply_fx_chain(samples, fx_config):
    """Apply a chain of post-processing effects."""
    out = samples
    if fx_config.get("reverse", False):
        out = apply_reverse(out)
    if fx_config.get("highpass", 0) > 0:
        out = apply_highpass(out, fx_config["highpass"])
    if fx_config.get("lowpass", 20000) < 20000:
        out = apply_lowpass(out, fx_config["lowpass"])
    if fx_config.get("eq_low", 0) != 0 or fx_config.get("eq_mid", 0) != 0 or fx_config.get("eq_high", 0) != 0:
        out = apply_eq(out, fx_config.get("eq_low", 0), fx_config.get("eq_mid", 0), fx_config.get("eq_high", 0))
    if fx_config.get("bitcrusher_bits", 0) > 0:
        out = apply_bitcrusher(out, fx_config["bitcrusher_bits"], fx_config.get("bitcrusher_sr", 1.0))
    if fx_config.get("distortion", 0) > 0:
        out = apply_distortion(out, fx_config["distortion"])
    if fx_config.get("tremolo_rate", 0) > 0:
        out = apply_tremolo(out, fx_config["tremolo_rate"], fx_config.get("tremolo_depth", 0.5))
    if fx_config.get("flanger_rate", 0) > 0:
        out = apply_flanger(out, fx_config["flanger_rate"], fx_config.get("flanger_depth", 3),
                            fx_config.get("flanger_feedback", 0.5), fx_config.get("flanger_mix", 0.5))
    if fx_config.get("phaser_rate", 0) > 0:
        out = apply_phaser(out, fx_config["phaser_rate"], fx_config.get("phaser_depth", 0.5),
                           fx_config.get("phaser_feedback", 0.3), fx_config.get("phaser_mix", 0.5))
    if fx_config.get("chorus_rate", 0) > 0:
        out = apply_chorus(out, fx_config["chorus_rate"], fx_config.get("chorus_depth", 7), fx_config.get("chorus_mix", 0.3))
    if fx_config.get("compressor_threshold", -60) > -60:
        out = apply_compressor(out, fx_config["compressor_threshold"], fx_config.get("compressor_ratio", 4))
    if fx_config.get("delay_ms", 0) > 0:
        out = apply_delay(out, fx_config["delay_ms"],
                          fx_config.get("delay_feedback", 0.3),
                          fx_config.get("delay_mix", 0.3))
    if fx_config.get("reverb", 0) > 0:
        out = apply_reverb(out, fx_config["reverb"])
    if fx_config.get("pitch_semitones", 0) != 0:
        out = apply_pitch_shift(out, fx_config["pitch_semitones"])
    # Normalization
    norm_mode = fx_config.get("normalize_mode", "off")
    if norm_mode != "off":
        target = fx_config.get("normalize_target", -3.0)
        out = normalize_audio(out, mode=norm_mode, target_db=target)
    return _to_mono_float(out)


# -- SFX Definitions ---------------------------------------------------------

SFX_DEFS = [
    {
        "name": "Laser / Blaster", "id": "laser", "category": "Sci-Fi",
        "params": [
            ("Frequency Start (Hz)", "freq_start", 1200, 200, 4000),
            ("Frequency End (Hz)", "freq_end", 200, 50, 2000),
            ("Duration (ms)", "duration", 300, 30, 2000),
            ("Volume", "volume", 80, 0, 100),
        ],
    },
    {
        "name": "Explosion", "id": "explosion", "category": "Impact",
        "params": [
            ("Duration (ms)", "duration", 800, 100, 3000),
            ("Low Freq (Hz)", "low_freq", 80, 20, 300),
            ("Noise Mix (%)", "noise_mix", 60, 0, 100),
            ("Decay (ms)", "decay", 600, 50, 2500),
            ("Volume", "volume", 85, 0, 100),
        ],
    },
    {
        "name": "Jump", "id": "jump", "category": "Game",
        "params": [
            ("Frequency Start (Hz)", "freq_start", 250, 80, 1000),
            ("Frequency Peak (Hz)", "freq_peak", 600, 200, 2000),
            ("Duration (ms)", "duration", 250, 30, 1000),
            ("Volume", "volume", 75, 0, 100),
        ],
    },
    {
        "name": "Coin / Pickup", "id": "coin", "category": "Game",
        "params": [
            ("Note 1 Freq (Hz)", "freq1", 988, 200, 3000),
            ("Note 2 Freq (Hz)", "freq2", 1319, 200, 4000),
            ("Note Duration (ms)", "note_dur", 80, 20, 500),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Power Up", "id": "powerup", "category": "Game",
        "params": [
            ("Freq Start (Hz)", "freq_start", 200, 50, 800),
            ("Freq End (Hz)", "freq_end", 1200, 200, 4000),
            ("Duration (ms)", "duration", 700, 100, 3000),
            ("Steps", "steps", 6, 2, 20),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Hit / Impact", "id": "hit", "category": "Impact",
        "params": [
            ("Duration (ms)", "duration", 200, 20, 1000),
            ("Noise Mix (%)", "noise_mix", 40, 0, 100),
            ("Pitch (Hz)", "pitch", 150, 30, 600),
            ("Volume", "volume", 90, 0, 100),
        ],
    },
    {
        "name": "Siren / Alarm", "id": "siren", "category": "Alert",
        "params": [
            ("Freq Low (Hz)", "freq_low", 400, 100, 1000),
            ("Freq High (Hz)", "freq_high", 900, 200, 3000),
            ("Duration (ms)", "duration", 1500, 200, 5000),
            ("Sweep Speed (Hz/s)", "sweep_speed", 4, 1, 20),
            ("Volume", "volume", 65, 0, 100),
        ],
    },
    {
        "name": "Whoosh", "id": "whoosh", "category": "Movement",
        "params": [
            ("Duration (ms)", "duration", 400, 50, 2000),
            ("Filter Freq Start (Hz)", "freq_start", 200, 50, 1000),
            ("Filter Freq End (Hz)", "freq_end", 4000, 500, 10000),
            ("Noise Mix (%)", "noise_mix", 90, 0, 100),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Beep", "id": "beep", "category": "UI",
        "params": [
            ("Frequency (Hz)", "frequency", 1000, 100, 5000),
            ("Duration (ms)", "duration", 150, 10, 2000),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Rain / Ambience", "id": "rain", "category": "Ambient",
        "params": [
            ("Duration (ms)", "duration", 3000, 500, 10000),
            ("Drop Density (%)", "density", 70, 10, 100),
            ("Low-pass Freq (Hz)", "lp_freq", 3000, 500, 10000),
            ("Volume", "volume", 50, 0, 100),
        ],
    },
    {
        "name": "Sweep (Rising)", "id": "sweep_up", "category": "Sci-Fi",
        "params": [
            ("Freq Start (Hz)", "freq_start", 100, 20, 2000),
            ("Freq End (Hz)", "freq_end", 2000, 100, 8000),
            ("Duration (ms)", "duration", 500, 50, 3000),
            ("Waveform", "waveform", 0, 0, 3),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Sweep (Falling)", "id": "sweep_down", "category": "Sci-Fi",
        "params": [
            ("Freq Start (Hz)", "freq_start", 2000, 100, 8000),
            ("Freq End (Hz)", "freq_end", 100, 20, 2000),
            ("Duration (ms)", "duration", 500, 50, 3000),
            ("Waveform", "waveform", 0, 0, 3),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Electric Zap", "id": "zap", "category": "Sci-Fi",
        "params": [
            ("Duration (ms)", "duration", 200, 20, 1000),
            ("Base Freq (Hz)", "base_freq", 800, 100, 5000),
            ("Discharges", "discharges", 8, 1, 30),
            ("Noise Mix (%)", "noise_mix", 50, 0, 100),
            ("Volume", "volume", 80, 0, 100),
        ],
    },
    {
        "name": "Sword Slash", "id": "slash", "category": "Combat",
        "params": [
            ("Duration (ms)", "duration", 300, 50, 1000),
            ("Swoosh Freq (Hz)", "freq", 2000, 500, 8000),
            ("Sharpness", "sharpness", 50, 0, 100),
            ("Volume", "volume", 80, 0, 100),
        ],
    },
    {
        "name": "UI Click", "id": "ui_click", "category": "UI",
        "params": [
            ("Frequency (Hz)", "frequency", 1200, 300, 5000),
            ("Duration (ms)", "duration", 50, 10, 200),
            ("Volume", "volume", 60, 0, 100),
        ],
    },
    {
        "name": "Bass Drop", "id": "bass_drop", "category": "Impact",
        "params": [
            ("Duration (ms)", "duration", 1200, 200, 4000),
            ("Start Freq (Hz)", "freq_start", 500, 100, 2000),
            ("End Freq (Hz)", "freq_end", 30, 20, 100),
            ("Distortion", "distortion", 1.5, 0, 5),
            ("Volume", "volume", 85, 0, 100),
        ],
    },
    {
        "name": "Glitch", "id": "glitch", "category": "Sci-Fi",
        "params": [
            ("Duration (ms)", "duration", 500, 50, 3000),
            ("Base Freq (Hz)", "base_freq", 440, 50, 2000),
            ("Stutter Rate (Hz)", "stutter_rate", 12, 1, 60),
            ("Bitcrush", "bitcrush", 6, 1, 16),
            ("Volume", "volume", 75, 0, 100),
        ],
    },
    {
        "name": "Menu Select", "id": "menu_select", "category": "UI",
        "params": [
            ("Freq 1 (Hz)", "freq1", 523, 200, 2000),
            ("Freq 2 (Hz)", "freq2", 659, 200, 3000),
            ("Freq 3 (Hz)", "freq3", 784, 200, 4000),
            ("Note Gap (ms)", "gap", 60, 10, 300),
            ("Volume", "volume", 65, 0, 100),
        ],
    },
    {
        "name": "Heartbeat", "id": "heartbeat", "category": "Ambient",
        "params": [
            ("Duration (ms)", "duration", 2000, 500, 5000),
            ("BPM", "bpm", 72, 40, 180),
            ("Low Thump Freq (Hz)", "thump_freq", 55, 20, 120),
            ("Volume", "volume", 75, 0, 100),
        ],
    },
    {
        "name": "Teleport", "id": "teleport", "category": "Sci-Fi",
        "params": [
            ("Duration (ms)", "duration", 800, 100, 3000),
            ("Freq Start (Hz)", "freq_start", 200, 50, 1000),
            ("Freq End (Hz)", "freq_end", 3000, 500, 8000),
            ("Sparkle Mix (%)", "sparkle", 40, 0, 100),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Chiptune Arpeggio", "id": "chiptune_arp", "category": "Music",
        "params": [
            ("Duration (ms)", "duration", 1000, 100, 5000),
            ("Base Note (Hz)", "base_note", 262, 50, 1000),
            ("Arp Speed (Hz)", "arp_speed", 8, 2, 30),
            ("Notes in Arp", "n_notes", 4, 2, 8),
            ("Volume", "volume", 65, 0, 100),
        ],
    },
    {
        "name": "Alarm Beep", "id": "alarm_beep", "category": "Alert",
        "params": [
            ("Duration (ms)", "duration", 2000, 200, 5000),
            ("Beep Freq (Hz)", "frequency", 1000, 200, 4000),
            ("Beep Rate (Hz)", "beep_rate", 3, 1, 15),
            ("Duty Cycle (%)", "duty", 50, 10, 90),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    # -- 8 NEW SFX types ------------------------------------------------
    {
        "name": "Glass Break", "id": "glass_break", "category": "Impact",
        "params": [
            ("Duration (ms)", "duration", 400, 50, 2000),
            ("Shatter Freq (Hz)", "freq", 3500, 1000, 10000),
            ("Fragments", "fragments", 15, 3, 50),
            ("Noise Mix (%)", "noise_mix", 50, 0, 100),
            ("Volume", "volume", 80, 0, 100),
        ],
    },
    {
        "name": "Water Splash", "id": "water_splash", "category": "Nature",
        "params": [
            ("Duration (ms)", "duration", 600, 100, 3000),
            ("Bubble Freq (Hz)", "bubble_freq", 300, 50, 1000),
            ("Splash Intensity (%)", "intensity", 70, 10, 100),
            ("Volume", "volume", 75, 0, 100),
        ],
    },
    {
        "name": "Wind", "id": "wind", "category": "Nature",
        "params": [
            ("Duration (ms)", "duration", 3000, 500, 10000),
            ("Gust Speed (Hz)", "gust_speed", 0.3, 0.05, 2.0),
            ("High-pass Freq (Hz)", "hp_freq", 200, 50, 2000),
            ("Volume", "volume", 60, 0, 100),
        ],
    },
    {
        "name": "Footsteps", "id": "footsteps", "category": "Movement",
        "params": [
            ("Duration (ms)", "duration", 2000, 500, 5000),
            ("Step Rate (Hz)", "step_rate", 2, 0.5, 6),
            ("Surface Type", "surface", 0, 0, 3),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Portal Open", "id": "portal", "category": "Sci-Fi",
        "params": [
            ("Duration (ms)", "duration", 1500, 300, 5000),
            ("Resonant Freq (Hz)", "res_freq", 200, 50, 800),
            ("Wobble Depth (%)", "wobble", 60, 0, 100),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Fanfare", "id": "fanfare", "category": "Music",
        "params": [
            ("Duration (ms)", "duration", 1500, 300, 5000),
            ("Base Freq (Hz)", "base_freq", 262, 100, 600),
            ("Brilliance (%)", "brilliance", 50, 0, 100),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Error Buzz", "id": "error_buzz", "category": "UI",
        "params": [
            ("Duration (ms)", "duration", 400, 50, 2000),
            ("Buzz Freq (Hz)", "frequency", 220, 50, 1000),
            ("Pulse Rate (Hz)", "pulse_rate", 6, 1, 20),
            ("Roughness (%)", "roughness", 40, 0, 100),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Metal Clang", "id": "metal_clang", "category": "Impact",
        "params": [
            ("Duration (ms)", "duration", 1000, 100, 4000),
            ("Resonant Freq (Hz)", "freq", 800, 200, 4000),
            ("Harmonics", "harmonics", 5, 1, 12),
            ("Decay (ms)", "decay", 500, 50, 3000),
            ("Volume", "volume", 80, 0, 100),
        ],
    },
    # -- 10 NEW SFX types (v4) -----------------------------------------
    {
        "name": "Vinyl Crackle", "id": "vinyl_crackle", "category": "Ambient",
        "params": [
            ("Duration (ms)", "duration", 3000, 500, 10000),
            ("Crackle Density (%)", "density", 40, 5, 100),
            ("Crackle Tone (Hz)", "tone_freq", 2000, 500, 8000),
            ("Volume", "volume", 50, 0, 100),
        ],
    },
    {
        "name": "Thunder", "id": "thunder", "category": "Nature",
        "params": [
            ("Duration (ms)", "duration", 4000, 1000, 10000),
            ("Rumble Freq (Hz)", "rumble_freq", 50, 20, 150),
            ("Crack Intensity (%)", "crack_intensity", 60, 0, 100),
            ("Decay (ms)", "decay", 3000, 500, 8000),
            ("Volume", "volume", 80, 0, 100),
        ],
    },
    {
        "name": "Drone Pad", "id": "drone", "category": "Ambient",
        "params": [
            ("Duration (ms)", "duration", 5000, 1000, 15000),
            ("Base Freq (Hz)", "base_freq", 110, 30, 400),
            ("Detune (%)", "detune", 5, 0, 30),
            ("Harmonics", "harmonics", 4, 1, 8),
            ("Volume", "volume", 60, 0, 100),
        ],
    },
    {
        "name": "Chime / Bell", "id": "chime", "category": "Music",
        "params": [
            ("Duration (ms)", "duration", 2000, 200, 5000),
            ("Fundamental (Hz)", "fundamental", 523, 100, 3000),
            ("Brightness (%)", "brightness", 60, 0, 100),
            ("Attack (ms)", "attack", 5, 1, 50),
            ("Volume", "volume", 75, 0, 100),
        ],
    },
    {
        "name": "Cinematic Riser", "id": "riser", "category": "Sci-Fi",
        "params": [
            ("Duration (ms)", "duration", 3000, 500, 10000),
            ("Start Freq (Hz)", "freq_start", 100, 20, 500),
            ("End Freq (Hz)", "freq_end", 4000, 1000, 12000),
            ("Noise Mix (%)", "noise_mix", 30, 0, 100),
            ("Volume", "volume", 75, 0, 100),
        ],
    },
    {
        "name": "Sub Drop", "id": "sub_drop", "category": "Impact",
        "params": [
            ("Duration (ms)", "duration", 2000, 500, 5000),
            ("Start Freq (Hz)", "freq_start", 200, 50, 500),
            ("End Freq (Hz)", "freq_end", 20, 10, 60),
            ("Distortion", "distortion", 0.5, 0, 3),
            ("Volume", "volume", 90, 0, 100),
        ],
    },
    {
        "name": "Digital Noise", "id": "digital_noise", "category": "Sci-Fi",
        "params": [
            ("Duration (ms)", "duration", 1000, 100, 5000),
            ("Bit Depth", "bit_depth", 4, 1, 12),
            ("Sample Rate Div", "sr_div", 8, 2, 44),
            ("Volume", "volume", 65, 0, 100),
        ],
    },
    {
        "name": "Crowd Cheer", "id": "crowd_cheer", "category": "Ambient",
        "params": [
            ("Duration (ms)", "duration", 3000, 500, 10000),
            ("Crowd Size (%)", "size", 60, 10, 100),
            ("Excitement (%)", "excitement", 70, 10, 100),
            ("Volume", "volume", 70, 0, 100),
        ],
    },
    {
        "name": "Machine Gun", "id": "machinegun", "category": "Combat",
        "params": [
            ("Duration (ms)", "duration", 2000, 200, 5000),
            ("Fire Rate (Hz)", "fire_rate", 10, 2, 30),
            ("Bullet Tone (Hz)", "tone", 800, 200, 3000),
            ("Echo Mix (%)", "echo_mix", 30, 0, 80),
            ("Volume", "volume", 80, 0, 100),
        ],
    },
    {
        "name": "Alarm Clock", "id": "alarm_clock", "category": "Alert",
        "params": [
            ("Duration (ms)", "duration", 2000, 200, 8000),
            ("Beep Freq (Hz)", "frequency", 1200, 400, 3000),
            ("Beep Rate (Hz)", "beep_rate", 4, 1, 12),
            ("Wobble (%)", "wobble", 20, 0, 80),
            ("Volume", "volume", 75, 0, 100),
        ],
    },
]


# -- Per-SFX generators ------------------------------------------------------

def _gen_laser(p, rng):
    dur = p["duration"] / 1000.0; n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)
    freq = np.linspace(p["freq_start"], p["freq_end"], n)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    return np.sin(phase) * _envelope_adsr(n, a=0.005, d=dur*0.3, s=0.4, r=dur*0.3)


def _gen_explosion(p, rng):
    dur = p["duration"] / 1000.0; decay = p["decay"] / 1000.0; n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False)
    rumble = np.sin(2*np.pi*p["low_freq"]*t) + 0.5*np.sin(2*np.pi*p["low_freq"]*1.5*t)
    noise = _noise(n, rng); mix = p["noise_mix"] / 100.0
    samples = (1-mix)*rumble + mix*noise
    env = np.exp(-3.0*t/max(decay, 0.01))
    att = int(0.002*SAMPLE_RATE)
    if att > 0: env[:att] = np.linspace(0,1,att,endpoint=False)
    return samples * env


def _gen_jump(p, rng):
    dur = p["duration"]/1000.0; n = int(dur*SAMPLE_RATE); t = np.linspace(0,dur,n,endpoint=False)
    mid = n//3; freq = np.zeros(n)
    freq[:mid] = np.linspace(p["freq_start"],p["freq_peak"],mid)
    freq[mid:] = np.linspace(p["freq_peak"],p["freq_start"],n-mid)
    phase = 2*np.pi*np.cumsum(freq)/SAMPLE_RATE
    return np.sin(phase) * _envelope_adsr(n,a=0.01,d=dur*0.15,s=0.7,r=dur*0.4)


def _gen_coin(p, rng):
    nd = p["note_dur"]/1000.0; n1=int(nd*SAMPLE_RATE); n2=int(nd*0.8*SAMPLE_RATE); gap=int(0.01*SAMPLE_RATE)
    t1=np.linspace(0,nd,n1,endpoint=False); t2=np.linspace(0,nd*0.8,n2,endpoint=False)
    s1=np.sin(2*np.pi*p["freq1"]*t1)*_envelope_adsr(n1,a=0.005,d=nd*0.2,s=0.6,r=nd*0.3)
    s2=np.sin(2*np.pi*p["freq2"]*t2)*_envelope_adsr(n2,a=0.005,d=nd*0.15,s=0.5,r=nd*0.4)
    return np.concatenate([s1,np.zeros(gap),s2])


def _gen_powerup(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    freq=np.linspace(p["freq_start"],p["freq_end"],n)
    phase=2*np.pi*np.cumsum(freq)/SAMPLE_RATE; wave=np.sin(phase)+0.3*np.sin(phase*2)
    steps=int(p["steps"]); env=np.zeros(n); sl=n//steps
    for i in range(steps): env[i*sl:min(i*sl+sl,n)]=(i+1)/steps
    env=np.clip(env,0,1); k=int(0.01*SAMPLE_RATE)
    if k>1: env=np.convolve(env,np.ones(k)/k,mode="same")
    return wave*env


def _gen_hit(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    tone=np.sin(2*np.pi*p["pitch"]*t); noise=_noise(n,rng); mix=p["noise_mix"]/100.0
    samples=(1-mix)*tone+mix*noise; env=np.exp(-8*t/max(dur,0.01))
    att=int(0.001*SAMPLE_RATE)
    if att>0: env[:att]=np.linspace(0,1,att,endpoint=False)
    return samples*env


def _gen_siren(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    spd=p["sweep_speed"]
    freq=(p["freq_low"]+p["freq_high"])/2+(p["freq_high"]-p["freq_low"])/2*np.sin(2*np.pi*spd*t)
    phase=2*np.pi*np.cumsum(freq)/SAMPLE_RATE
    return np.sin(phase)*_envelope_adsr(n,a=dur*0.1,d=dur*0.05,s=0.8,r=dur*0.15)


def _gen_whoosh(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    noise=_noise(n,rng); mix=p["noise_mix"]/100.0
    env=np.sin(np.pi*t/dur)**0.5
    # Vectorized variable low-pass using uniform_filter1d
    cutoff_norm=np.linspace(p["freq_start"]/SAMPLE_RATE,p["freq_end"]/SAMPLE_RATE,n)
    cutoff_norm=np.clip(cutoff_norm,0.001,0.5)
    window_sizes=(1.0/(2*cutoff_norm)).astype(int)
    window_sizes=np.clip(window_sizes,1,n)
    # Process in chunks with similar window sizes for efficiency
    filtered = np.zeros(n)
    prev_ws = -1
    chunk_start = 0
    for i in range(n):
        ws = window_sizes[i]
        if ws != prev_ws or i == n - 1:
            if i > chunk_start and prev_ws > 0:
                end = i + 1
                filtered[chunk_start:end] = uniform_filter1d(noise[chunk_start:end], size=prev_ws, mode='nearest')
            chunk_start = i
            prev_ws = ws
    tone=np.sin(2*np.pi*np.linspace(p["freq_start"],p["freq_end"],n)*t/np.maximum(t,1e-6))
    return (mix*filtered+(1-mix)*tone*0.3)*env


def _gen_beep(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    return np.sin(2*np.pi*p["frequency"]*t)*_envelope_adsr(n,a=0.003,d=dur*0.1,s=0.8,r=dur*0.2)


def _gen_rain(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    density=p["density"]/100.0; noise=_noise(n,rng)
    nyq=SAMPLE_RATE/2; cut=min(p["lp_freq"],nyq-1)
    b,a=sig.butter(2,cut/nyq,btype='low'); noise=sig.lfilter(b,a,noise)
    num_drops=int(n*density*0.002); drops=np.zeros(n)
    for _ in range(num_drops):
        pos=rng.randint(0,max(n-200,1)); length=rng.randint(10,60)
        amp=rng.uniform(0.3,1.0); drop_env=np.exp(-np.arange(length)/(length*0.15))
        end=min(pos+length,n); actual=end-pos
        drops[pos:end]+=amp*drop_env[:actual]*rng.uniform(-1,1,actual)
    samples=noise*0.6+drops*0.4
    fade=int(0.1*SAMPLE_RATE)
    if 0<fade<n: samples[:fade]*=np.linspace(0,1,fade,endpoint=False); samples[-fade:]*=np.linspace(1,0,fade,endpoint=False)
    return samples


def _gen_sweep(p, rising=True, rng=None):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    fs=p["freq_start"] if rising else p["freq_end"]; fe=p["freq_end"] if rising else p["freq_start"]
    freq=np.linspace(fs,fe,n); phase=2*np.pi*np.cumsum(freq)/SAMPLE_RATE
    wi=int(p.get("waveform",0))
    if wi==1: samples=np.sign(np.sin(phase))
    elif wi==2: samples=2.0*(phase/(2*np.pi)-np.floor(0.5+phase/(2*np.pi)))
    elif wi==3: p2=phase/(2*np.pi); samples=2.0*np.abs(2.0*(p2-np.floor(p2+0.5)))-1.0
    else: samples=np.sin(phase)
    return samples*_envelope_adsr(n,a=0.01,d=dur*0.15,s=0.7,r=dur*0.2)

def _gen_sweep_up(p, rng): return _gen_sweep(p, True, rng)
def _gen_sweep_down(p, rng): return _gen_sweep(p, False, rng)


def _gen_zap(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    dis=int(p["discharges"]); mix=p["noise_mix"]/100.0; noise=_noise(n,rng)
    tone=np.sin(2*np.pi*p["base_freq"]*t); burst_env=np.zeros(n); sl=n//dis
    for i in range(dis): s=i*sl; burst_env[s:min(s+int(sl*0.6),n)]=1.0
    return ((1-mix)*tone+mix*noise)*burst_env*np.exp(-4*t/max(dur,0.01))


def _gen_slash(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    noise=_noise(n,rng); sharp=p["sharpness"]/100.0
    nyq=SAMPLE_RATE/2; cut=min(p["freq"],nyq-1)
    b,a=sig.butter(2,cut/nyq,btype='low'); filtered=sig.lfilter(b,a,noise)
    samples=sharp*noise+(1-sharp)*filtered; env=np.sin(np.pi*t/dur)**0.3
    att=int(0.005*SAMPLE_RATE)
    if 0<att<n: env[:att]=np.linspace(0,1,att,endpoint=False)
    return samples*env


def _gen_ui_click(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    samples=np.sin(2*np.pi*p["frequency"]*t)+0.3*np.sin(2*np.pi*p["frequency"]*2*t)
    env=np.exp(-12*t/max(dur,0.01)); att=int(0.001*SAMPLE_RATE)
    if att>0: env[:att]=np.linspace(0,1,att,endpoint=False)
    return samples*env


def _gen_bass_drop(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    freq=np.linspace(p["freq_start"],p["freq_end"],n); phase=2*np.pi*np.cumsum(freq)/SAMPLE_RATE
    samples=np.sin(phase)+0.5*np.sin(phase*2)
    rl=n//3; samples[-rl:]+=0.4*np.sin(2*np.pi*p["freq_end"]*t[-rl:])
    samples=np.tanh(samples*p["distortion"]); env=np.exp(-2.5*t/max(dur,0.01))
    att=int(0.003*SAMPLE_RATE)
    if att>0: env[:att]=np.linspace(0,1,att,endpoint=False)
    return samples*env


def _gen_glitch(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    base=np.sin(2*np.pi*p["base_freq"]*t); rate=p["stutter_rate"]; sl=int(SAMPLE_RATE/rate)
    out=np.zeros(n)
    for i in range(0,n,sl):
        end=min(i+sl,n); out[i:end]=base[i:end]*rng.uniform(0.5,2.0)
    levels=2**int(p["bitcrush"]); out=np.round(out*levels)/levels
    env=np.ones(n); fade=int(0.005*SAMPLE_RATE)
    if 0<fade<n: env[:fade]=np.linspace(0,1,fade,endpoint=False); env[-fade:]=np.linspace(1,0,fade,endpoint=False)
    return out*env


def _gen_menu_select(p, rng):
    nd=0.08; nn=int(nd*SAMPLE_RATE); gap=int(p["gap"]/1000.0*SAMPLE_RATE)
    env_n=_envelope_adsr(nn,a=0.005,d=nd*0.2,s=0.6,r=nd*0.3); parts=[]
    for i,freq in enumerate([p["freq1"],p["freq2"],p["freq3"]]):
        t=np.linspace(0,nd,nn,endpoint=False)
        parts.append(np.sin(2*np.pi*freq*(1+i*0.02)*t)*env_n)
        if i<2: parts.append(np.zeros(gap))
    return np.concatenate(parts)


def _gen_heartbeat(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    bp=60.0/p["bpm"]; samples=np.zeros(n)
    for bt in np.arange(0,dur,bp):
        bs=int(bt*SAMPLE_RATE)
        for to,td,ta in [(0.0,0.08,1.0),(0.12,0.06,0.7)]:
            ts=bs+int(to*SAMPLE_RATE); tn=int(td*SAMPLE_RATE)
            if ts+tn>n: continue
            tt=np.linspace(0,td,tn,endpoint=False)
            th=np.sin(2*np.pi*p["thump_freq"]*tt)+0.3*np.sin(2*np.pi*p["thump_freq"]*2*tt)
            samples[ts:ts+tn]+=th*np.exp(-15*tt/td)*ta
    return samples


def _gen_teleport(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    freq=np.linspace(p["freq_start"],p["freq_end"],n); phase=2*np.pi*np.cumsum(freq)/SAMPLE_RATE
    sweep=np.sin(phase)+0.3*np.sin(phase*0.5); sparkle=_noise(n,rng)
    nyq=SAMPLE_RATE/2; b,a=sig.butter(2,5000/nyq,btype='low'); sparkle=sig.lfilter(b,a,sparkle)
    sparkle*=np.sin(np.pi*t/dur)**2; mix=p["sparkle"]/100.0
    samples=(1-mix)*sweep+mix*sparkle; env=np.sin(np.pi*t/dur)**0.3
    fade=int(0.02*SAMPLE_RATE)
    if 0<fade<n: env[:fade]=np.linspace(0,1,fade,endpoint=False); env[-fade:]=np.linspace(1,0,fade,endpoint=False)
    return samples*env


def _gen_chiptune_arp(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    base=p["base_note"]; nn=int(p["n_notes"]); aspd=p["arp_speed"]
    intervals=[1.0,1.25,1.5,1.667,2.0,2.5,3.0,3.333][:nn]
    sl=int(SAMPLE_RATE/aspd); fa=np.zeros(n)
    for i in range(n): fa[i]=base*intervals[(i//sl)%len(intervals)]
    phase=2*np.pi*np.cumsum(fa)/SAMPLE_RATE
    samples=np.sign(np.sin(phase))*0.6+np.sin(phase)*0.4
    return samples*_envelope_adsr(n,a=0.005,d=dur*0.05,s=0.8,r=dur*0.15)


def _gen_alarm_beep(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    gp=1.0/p["beep_rate"]; gate=((t%gp)/gp)<(p["duty"]/100.0)
    return np.sin(2*np.pi*p["frequency"]*t)*gate.astype(np.float64)*_envelope_adsr(n,a=dur*0.05,d=0,s=1.0,r=dur*0.1)


# -- 8 NEW generators ------------------------------------------------------

def _gen_glass_break(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    mix=p["noise_mix"]/100.0; noise=_noise(n,rng)
    nyq=SAMPLE_RATE/2; f_lo=max(p["freq"]*0.5,1)/nyq; f_hi=min(p["freq"]*2.0,nyq-1)/nyq
    if f_lo < f_hi: b,a=sig.butter(2,[f_lo,f_hi],btype='band'); filtered=sig.lfilter(b,a,noise)
    else: filtered=noise
    # Add random shatter pings
    shatter=np.zeros(n)
    frags=int(p["fragments"])
    for _ in range(frags):
        pos=rng.randint(0,max(n-500,1)); flen=rng.randint(50,300)
        freq_mult=rng.uniform(0.7,1.4)
        ft=np.linspace(0,flen/SAMPLE_RATE,flen,endpoint=False)
        ping=np.sin(2*np.pi*p["freq"]*freq_mult*ft)*np.exp(-20*ft)
        end=min(pos+flen,n); actual=end-pos
        shatter[pos:end]+=ping[:actual]*rng.uniform(0.3,1.0)
    samples=(1-mix)*shatter+mix*filtered; env=np.exp(-6*t/max(dur,0.01))
    att=int(0.001*SAMPLE_RATE)
    if att>0: env[:att]=np.linspace(0,1,att,endpoint=False)
    return samples*env


def _gen_water_splash(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    intensity=p["intensity"]/100.0; noise=_noise(n,rng)
    # Low-pass for water feel
    nyq=SAMPLE_RATE/2; b,a=sig.butter(3,min(2000,nyq-1)/nyq,btype='low')
    water=sig.lfilter(b,a,noise)
    # Bubbles
    num_bubbles=int(n*intensity*0.003); bubbles=np.zeros(n)
    for _ in range(num_bubbles):
        pos=rng.randint(0,max(n-300,1)); blen=rng.randint(20,100)
        bf=p["bubble_freq"]*rng.uniform(0.5,2.0)
        bt=np.linspace(0,blen/SAMPLE_RATE,blen,endpoint=False)
        bubble=np.sin(2*np.pi*bf*bt)*np.exp(-8*bt/(blen/SAMPLE_RATE))
        end=min(pos+blen,n); bubbles[pos:end]+=bubble[:end-pos]*rng.uniform(0.2,0.8)
    samples=water*0.5+bubbles*0.5; env=np.exp(-3*t/max(dur*0.5,0.01))
    att=int(0.003*SAMPLE_RATE)
    if att>0: env[:att]=np.linspace(0,1,att,endpoint=False)
    return samples*env


def _gen_wind(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    noise=_noise(n,rng)
    # Slow amplitude modulation for gusts
    gust_mod=0.5+0.5*np.sin(2*np.pi*p["gust_speed"]*t)
    # Additional random gusts
    gust_env=np.zeros(n)
    num_gusts=rng.randint(3,10)
    for _ in range(num_gusts):
        pos=rng.randint(0,max(n-1,1)); glen=rng.randint(int(0.1*SAMPLE_RATE),int(0.5*SAMPLE_RATE))
        end=min(pos+glen,n); gust_env[pos:end]+=np.sin(np.pi*np.arange(end-pos)/glen)
    samples=noise*gust_mod*(0.5+0.5*gust_env)
    # High-pass to remove rumble
    nyq=SAMPLE_RATE/2; b,a=sig.butter(2,min(p["hp_freq"],nyq-1)/nyq,btype='high')
    samples=sig.lfilter(b,a,samples)
    fade=int(0.2*SAMPLE_RATE)
    if 0<fade<n: samples[:fade]*=np.linspace(0,1,fade,endpoint=False); samples[-fade:]*=np.linspace(1,0,fade,endpoint=False)
    return samples


def _gen_footsteps(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    step_period=1.0/p["step_rate"]; surface=int(p["surface"])
    # Surface characteristics: 0=dirt, 1=stone, 2=wood, 3=metal
    hp_freqs=[800, 2000, 1500, 4000]; lp_freqs=[2000, 6000, 4000, 8000]
    noise_tone_mix=[0.7, 0.3, 0.5, 0.2]; decay_times=[0.05, 0.08, 0.04, 0.12]
    hp=hp_freqs[surface]; lp=lp_freqs[surface]; nm=noise_tone_mix[surface]; dt=decay_times[surface]
    samples=np.zeros(n); step_freq=80+surface*40
    for st in np.arange(0, dur, step_period):
        si=int(st*SAMPLE_RATE); sn=int(dt*SAMPLE_RATE)
        if si+sn>n: break
        st2=np.linspace(0,dt,sn,endpoint=False)
        tone=np.sin(2*np.pi*step_freq*st2)+0.3*np.sin(2*np.pi*step_freq*2.5*st2)
        ns=_noise(sn,rng)
        nyq=SAMPLE_RATE/2
        b1,a1=sig.butter(2,min(hp,nyq-1)/nyq,btype='high'); ns=sig.lfilter(b1,a1,ns)
        step=(1-nm)*tone+nm*ns
        step*=np.exp(-15*st2/dt)*rng.uniform(0.6,1.0)
        samples[si:si+sn]+=step
    return samples


def _gen_portal(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    rf=p["res_freq"]; wb=p["wobble"]/100.0
    # Modulated resonant frequency with wobble
    mod=rf+rf*wb*np.sin(2*np.pi*2.0*t)
    phase=2*np.pi*np.cumsum(mod)/SAMPLE_RATE
    core=np.sin(phase)+0.4*np.sin(phase*1.5)+0.2*np.sin(phase*2.0)
    # Add ethereal pad
    pad_freq=rf*0.5; pad=np.sin(2*np.pi*pad_freq*t)*0.3
    # Noise shimmer
    shimmer=_noise(n,rng)*0.15
    nyq=SAMPLE_RATE/2; b,a=sig.butter(2,3000/nyq,btype='low'); shimmer=sig.lfilter(b,a,shimmer)
    samples=core+pad+shimmer; env=np.sin(np.pi*t/dur)**0.5
    fade=int(0.05*SAMPLE_RATE)
    if 0<fade<n: env[:fade]=np.linspace(0,1,fade,endpoint=False); env[-fade:]=np.linspace(1,0,fade,endpoint=False)
    return samples*env


def _gen_fanfare(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    bf=p["base_freq"]; br=p["brilliance"]/100.0
    # Three-note major chord arpeggio with harmonics
    notes=[1.0, 1.25, 1.5]  # root, major third, fifth
    note_dur=dur/3; samples=np.zeros(n)
    for i,nr in enumerate(notes):
        start=int(i*note_dur*SAMPLE_RATE); end=int((i+1)*note_dur*SAMPLE_RATE)
        end=min(end,n); nl=end-start
        nt=np.linspace(0,nl/SAMPLE_RATE,nl,endpoint=False)
        freq=bf*nr
        note=np.sin(2*np.pi*freq*nt)
        note+=br*0.5*np.sin(2*np.pi*freq*2*nt)  # 2nd harmonic
        note+=br*0.3*np.sin(2*np.pi*freq*3*nt)  # 3rd harmonic
        note+=br*0.15*np.sin(2*np.pi*freq*4*nt)  # 4th harmonic
        env_n=np.ones(nl)
        fade_in=int(0.01*SAMPLE_RATE); fade_out=int(0.03*SAMPLE_RATE)
        if fade_in>0 and fade_in<nl: env_n[:fade_in]=np.linspace(0,1,fade_in,endpoint=False)
        if fade_out>0 and fade_out<nl: env_n[-fade_out:]=np.linspace(1,0.8,fade_out,endpoint=False)
        samples[start:end]+=note*env_n
    return samples


def _gen_error_buzz(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    rate=p["pulse_rate"]; rough=p["roughness"]/100.0
    # Pulsed square-ish wave
    pulse=0.5+0.5*np.sign(np.sin(2*np.pi*rate*t))
    buzz=np.sign(np.sin(2*np.pi*p["frequency"]*t))*0.7+np.sin(2*np.pi*p["frequency"]*t)*0.3
    # Add roughness (harsh noise bursts)
    harsh=_noise(n,rng)*rough
    nyq=SAMPLE_RATE/2; b,a=sig.butter(2,3000/nyq,btype='low'); harsh=sig.lfilter(b,a,harsh)
    samples=buzz*pulse+harsh*0.4
    env=_envelope_adsr(n,a=dur*0.02,d=0,s=1.0,r=dur*0.1)
    return samples*env


def _gen_metal_clang(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    decay=p["decay"]/1000.0; num_h=int(p["harmonics"]); freq=p["freq"]
    samples=np.zeros(n)
    for h in range(1, num_h+1):
        h_freq=freq*h
        # Slightly detuned harmonics for metallic character
        detune=1.0+rng.uniform(-0.01,0.01)
        amp=1.0/(h**0.7)  # Higher harmonics decay faster
        harmonic=np.sin(2*np.pi*h_freq*detune*t)*amp
        # Per-harmonic decay
        h_decay=decay*(1.0-0.5*h/num_h)
        h_env=np.exp(-4*t/max(h_decay,0.01))
        samples+=harmonic*h_env
    # Add initial impact noise
    impact=_noise(n,rng)*0.3*np.exp(-20*t/0.01)
    nyq=SAMPLE_RATE/2; f_lo=max(freq*1.5,1)/nyq; f_hi=min(freq*3.0,nyq-1)/nyq
    if f_lo < f_hi: b,a=sig.butter(2,[f_lo,f_hi],btype='band'); impact=sig.lfilter(b,a,impact)
    samples+=impact
    return samples


# -- 10 NEW generators (v4) ----------------------------------------------

def _gen_vinyl_crackle(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    density=p["density"]/100.0; tf=p["tone_freq"]
    noise=_noise(n,rng)
    nyq=SAMPLE_RATE/2
    b_hiss,a_hiss=sig.butter(2,min(tf,nyq-1)/nyq,btype='low')
    hiss=sig.lfilter(b_hiss,a_hiss,noise)*0.15
    # Vectorized crackle: generate all pops at once using np.add.at
    num_pops=int(n*density*0.0004)
    if num_pops > 0:
        positions = rng.randint(0, max(n-1, 1), num_pops)
        pop_lens = rng.randint(5, 41, num_pops)
        amps = rng.uniform(0.3, 1.0, num_pops)
        max_pop = 40
        # Build pop envelope matrix (num_pops x max_pop)
        pop_range = np.arange(max_pop)
        pop_envs = np.exp(-pop_range / (pop_range * 0.2 + 1e-6))  # avoid div by 0
        # For each pop, scatter-add its envelope
        crackle = np.zeros(n)
        for i in range(num_pops):
            pos = positions[i]
            pl = min(pop_lens[i], max_pop)
            end = min(pos + pl, n)
            actual = end - pos
            if actual > 0:
                crackle[pos:end] += amps[i] * pop_envs[:actual] * rng.uniform(-1, 1, actual)
    else:
        crackle = np.zeros(n)
    groove_mod=0.3+0.2*np.sin(2*np.pi*0.8*t)
    samples=(hiss+crackle*0.7)*groove_mod
    fade=int(0.05*SAMPLE_RATE)
    if 0<fade<n: samples[:fade]*=np.linspace(0,1,fade,endpoint=False); samples[-fade:]*=np.linspace(1,0,fade,endpoint=False)
    return samples


def _gen_thunder(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    decay=p["decay"]/1000.0; crack=p["crack_intensity"]/100.0; rf=p["rumble_freq"]
    crack_noise=_noise(n,rng)
    nyq=SAMPLE_RATE/2
    b_crack,a_crack=sig.butter(2,min(6000,nyq-1)/nyq,btype='low')
    crack_filtered=sig.lfilter(b_crack,a_crack,crack_noise)
    crack_env=np.exp(-t/0.05)*crack
    rumble_noise=_noise(n,rng)
    b_rumble,a_rumble=sig.butter(3,min(200,nyq-1)/nyq,btype='low')
    rumble=sig.lfilter(b_rumble,a_rumble,rumble_noise)
    rumble+=np.sin(2*np.pi*rf*t)*0.5+np.sin(2*np.pi*rf*1.5*t)*0.3
    rumble_env=np.exp(-t/max(decay,0.01))
    rumble2=np.sin(2*np.pi*(rf*0.7)*t)*0.3+np.sin(2*np.pi*(rf*2.1)*t)*0.15
    rumble2*=np.exp(-t/max(decay*1.5,0.01))*0.5
    extra_rumble=np.zeros(n)
    for _ in range(rng.randint(2,5)):
        offset=rng.randint(0,max(int(0.3*SAMPLE_RATE),1))
        r_dur=rng.uniform(0.3,0.8)
        r_n=min(int(r_dur*SAMPLE_RATE), n-offset)
        if r_n>10:
            r_t=np.linspace(0,r_dur,r_n,endpoint=False)
            r_sig=np.sin(2*np.pi*(rf*rng.uniform(0.6,1.4))*r_t)*rng.uniform(0.1,0.4)
            r_sig*=np.exp(-r_t/r_dur*2)
            extra_rumble[offset:offset+r_n]+=r_sig
    samples=crack_filtered*crack_env+(rumble+rumble2)*rumble_env*0.6+extra_rumble
    fade=int(0.01*SAMPLE_RATE)
    if 0<fade<n: samples[:fade]*=np.linspace(0,1,fade,endpoint=False)
    tail_fade=int(min(decay*0.5, dur*0.3)*SAMPLE_RATE)
    if tail_fade>0 and tail_fade<n: samples[-tail_fade:]*=np.linspace(1,0,tail_fade,endpoint=False)
    return samples


def _gen_drone(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    bf=p["base_freq"]; det=p["detune"]/100.0; num_h=int(p["harmonics"])
    samples=np.zeros(n)
    for voice in range(3):
        detune_factor=1.0+(voice-1)*det*0.01
        for h in range(1, num_h+1):
            h_freq=bf*h*detune_factor*(1.0+rng.uniform(-0.002,0.002))
            amp=1.0/(h**0.8)
            mod_freq=rng.uniform(0.05,0.3)
            mod_depth=rng.uniform(0.05,0.2)
            lfo=1.0-mod_depth+mod_depth*np.sin(2*np.pi*mod_freq*t+voice*0.5)
            # Use cumsum-based phase for numerical stability
            phase=2*np.pi*np.cumsum(np.full(n,h_freq))/SAMPLE_RATE
            samples+=np.sin(phase)*amp*lfo
    noise=_noise(n,rng)*0.05
    nyq=SAMPLE_RATE/2
    b,a=sig.butter(2,min(2000,nyq-1)/nyq,btype='low')
    noise=sig.lfilter(b,a,noise)
    samples+=noise
    fade=int(0.3*SAMPLE_RATE)
    if 0<fade<n: samples[:fade]*=np.linspace(0,1,fade,endpoint=False); samples[-fade:]*=np.linspace(1,0,fade,endpoint=False)
    return samples


def _gen_chime(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    fund=p["fundamental"]; bright=p["brightness"]/100.0; att=p["attack"]/1000.0
    partials=[1.0, 2.0, 3.01, 4.03, 5.04, 6.07, 7.11, 8.15]
    amplitudes=[1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.07, 0.04]
    decay_rates=[1.0, 1.5, 2.2, 3.0, 4.0, 5.0, 6.5, 8.0]
    samples=np.zeros(n)
    for i,(pr,amp,dr) in enumerate(zip(partials,amplitudes,decay_rates)):
        freq=fund*pr
        if i >= 2: amp*=bright
        detune=1.0+rng.uniform(-0.001,0.001)
        samples+=np.sin(2*np.pi*freq*detune*t)*amp*np.exp(-dr*t/max(dur,0.01))
    env=np.ones(n)
    att_n=int(att*SAMPLE_RATE)
    if att_n>0 and att_n<n: env[:att_n]=np.linspace(0,1,att_n,endpoint=False)
    env*=np.exp(-2.0*t/max(dur*0.6,0.01))
    return samples*env


def _gen_riser(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    fs=p["freq_start"]; fe=p["freq_end"]; nm=p["noise_mix"]/100.0
    freq=np.exp(np.linspace(np.log(fs),np.log(max(fe,fs+1)),n))
    phase=2*np.pi*np.cumsum(freq)/SAMPLE_RATE
    tone=np.sin(phase)+0.5*np.sin(phase*2)+0.3*np.sin(phase*3)
    noise=_noise(n,rng)
    noise_env=np.linspace(0,1,n)**2
    nyq=SAMPLE_RATE/2
    b,a=sig.butter(2,min(fe,nyq-1)/nyq,btype='low')
    noise_filtered=sig.lfilter(b,a,noise)*noise_env
    tone_env=np.linspace(1,0,n)**0.5
    samples=(1-nm)*tone*tone_env+nm*noise_filtered
    amp_env=np.sin(np.pi*t/dur)**0.5
    return samples*amp_env


def _gen_sub_drop(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    fs=p["freq_start"]; fe=p["freq_end"]; dist=p["distortion"]
    freq=np.exp(np.linspace(np.log(fs),np.log(max(fe,10)),n))
    phase=2*np.pi*np.cumsum(freq)/SAMPLE_RATE
    samples=np.sin(phase)+0.6*np.sin(phase*0.5)+0.3*np.sin(phase*2)
    noise=_noise(n,rng)*0.15
    nyq=SAMPLE_RATE/2
    b,a=sig.butter(2,min(300,nyq-1)/nyq,btype='low')
    noise=sig.lfilter(b,a,noise)
    noise_env=np.exp(-t/max(dur*0.3,0.01))
    samples+=noise*noise_env
    if dist > 0: samples=np.tanh(samples*dist)
    env=np.exp(-2.0*t/max(dur,0.01))
    att=int(0.003*SAMPLE_RATE)
    if att>0: env[:att]=np.linspace(0,1,att,endpoint=False)
    tail_start=int(dur*0.4*SAMPLE_RATE)
    if tail_start < n:
        rumble=np.sin(2*np.pi*fe*t[tail_start:])*0.3
        rumble*=np.exp(-3*(t[tail_start:]-t[tail_start])/max(dur*0.6,0.01))
        samples[tail_start:]+=rumble
    return samples*env


def _gen_digital_noise(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    bd=int(p["bit_depth"]); sr_div=int(p["sr_div"])
    noise=_noise(n,rng)
    levels=2**bd
    crushed=np.round(noise*levels)/levels
    if sr_div > 1:
        step=sr_div
        downsampled=crushed[::step]
        orig_idx=np.linspace(0,len(downsampled)-1,n)
        int_idx=orig_idx.astype(int)
        frac_idx=orig_idx-int_idx
        int_idx=np.clip(int_idx,0,len(downsampled)-1)
        next_idx=np.clip(int_idx+1,0,len(downsampled)-1)
        crushed=downsampled[int_idx]*(1-frac_idx)+downsampled[next_idx]*frac_idx
    artifact_freq=SAMPLE_RATE/sr_div
    artifact=np.sin(2*np.pi*artifact_freq*t)*0.1
    samples=crushed+artifact
    fade=int(0.005*SAMPLE_RATE)
    if 0<fade<n: samples[:fade]*=np.linspace(0,1,fade,endpoint=False); samples[-fade:]*=np.linspace(1,0,fade,endpoint=False)
    return samples


def _gen_crowd_cheer(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    size=p["size"]/100.0; excite=p["excitement"]/100.0
    num_voices=max(int(size*30),5)
    samples=np.zeros(n)
    for _ in range(num_voices):
        voice_len=rng.randint(int(0.05*SAMPLE_RATE),int(0.5*SAMPLE_RATE))
        start_pos=rng.randint(0,max(n-voice_len,1))
        end=min(start_pos+voice_len,n)
        actual=end-start_pos
        v_noise=rng.uniform(-1,1,actual)
        formant=rng.uniform(300,3000)
        nyq=SAMPLE_RATE/2
        if formant < 1000:
            f_lo = max(formant * 0.5, 1) / nyq
            f_hi = min(formant * 1.5, nyq - 1) / nyq
            if f_lo < f_hi and f_lo > 0 and f_hi < 1:
                b,a=sig.butter(2,[f_lo,f_hi],btype='band')
            else:
                b,a=sig.butter(2,min(formant,nyq-1)/nyq,btype='low')
        else:
            b,a=sig.butter(2,min(formant,nyq-1)/nyq,btype='low')
        v_filtered=sig.lfilter(b,a,v_noise)
        v_env=np.ones(actual)
        v_att=int(0.01*SAMPLE_RATE)
        v_dec=int(rng.uniform(0.02,0.1)*SAMPLE_RATE)
        if v_att>0 and v_att<actual: v_env[:v_att]=np.linspace(0,1,v_att,endpoint=False)
        if v_dec>0 and v_dec<actual: v_env[-v_dec:]=np.linspace(1,0,v_dec,endpoint=False)
        samples[start_pos:end]+=v_filtered*v_env*rng.uniform(0.3,1.0)
    if excite > 0.3:
        clap_rate=2+excite*6
        clap_period=1.0/clap_rate
        for ct in np.arange(0,dur,clap_period):
            ci=int(ct*SAMPLE_RATE)
            cl=int(0.005*SAMPLE_RATE)
            if ci+cl<n:
                clap=_noise(cl,rng)*np.exp(-np.arange(cl)/(cl*0.3))*excite*0.3
                samples[ci:ci+cl]+=clap
    env=np.sin(np.pi*t/dur)**0.3
    fade=int(0.05*SAMPLE_RATE)
    if 0<fade<n: env[:fade]*=np.linspace(0,1,fade,endpoint=False); env[-fade:]*=np.linspace(1,0,fade,endpoint=False)
    return samples*env


def _gen_machinegun(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    rate=p["fire_rate"]; tone=p["tone"]; echo=p["echo_mix"]/100.0
    burst_period=1.0/rate; burst_len=int(0.015*SAMPLE_RATE)
    samples=np.zeros(n)
    burst_t=np.linspace(0,0.015,burst_len,endpoint=False)
    bullet=np.sin(2*np.pi*tone*burst_t)*np.exp(-burst_t/0.003)
    bullet+=_noise(burst_len,rng)*0.5*np.exp(-burst_t/0.005)
    for bt in np.arange(0, dur, burst_period):
        bi=int(bt*SAMPLE_RATE)
        end=min(bi+burst_len,n); actual=end-bi
        if actual>0:
            jitter=rng.randint(-int(0.002*SAMPLE_RATE),int(0.002*SAMPLE_RATE)+1)
            si=max(bi+jitter,0); se=min(si+actual,n); sa=se-si
            if sa>0:
                amp=rng.uniform(0.7,1.0)
                samples[si:se]+=bullet[:sa]*amp
                if echo>0:
                    echo_delay=int(0.05*SAMPLE_RATE)
                    echo_end=min(si+echo_delay+sa,n)
                    if echo_end>si+echo_delay:
                        echo_start=si+echo_delay
                        echo_actual=min(sa, echo_end-echo_start)
                        samples[echo_start:echo_start+echo_actual]+=bullet[:echo_actual]*echo*0.5
    rumble=_noise(n,rng)*0.15
    nyq=SAMPLE_RATE/2
    b,a=sig.butter(2,min(150,nyq-1)/nyq,btype='low')
    rumble=sig.lfilter(b,a,rumble)
    rumble_env=0.5+0.5*np.sign(np.sin(2*np.pi*rate*t))
    samples+=rumble*rumble_env
    fade=int(0.005*SAMPLE_RATE)
    if 0<fade<n: samples[:fade]*=np.linspace(0,1,fade,endpoint=False); samples[-fade:]*=np.linspace(1,0,fade,endpoint=False)
    return samples


def _gen_alarm_clock(p, rng):
    dur=p["duration"]/1000.0; n=int(dur*SAMPLE_RATE); t=np.linspace(0,dur,n,endpoint=False)
    freq=p["frequency"]; br=p["beep_rate"]; wb=p["wobble"]/100.0
    beep_period=1.0/br
    gate=((t%beep_period)/beep_period)<0.5
    wobble_freq=freq+freq*wb*np.sin(2*np.pi*6*t)
    phase=2*np.pi*np.cumsum(wobble_freq)/SAMPLE_RATE
    tone=np.sin(phase)
    tone+=0.3*np.sin(phase*2)+0.1*np.sin(phase*3)
    samples=tone*gate.astype(np.float64)
    env=_envelope_adsr(n,a=dur*0.01,d=0,s=1.0,r=dur*0.05)
    return samples*env


# -- Main API ----------------------------------------------------------------

_GENERATORS = {
    "laser": _gen_laser, "explosion": _gen_explosion, "jump": _gen_jump,
    "coin": _gen_coin, "powerup": _gen_powerup, "hit": _gen_hit,
    "siren": _gen_siren, "whoosh": _gen_whoosh, "beep": _gen_beep,
    "rain": _gen_rain, "sweep_up": _gen_sweep_up, "sweep_down": _gen_sweep_down,
    "zap": _gen_zap, "slash": _gen_slash, "ui_click": _gen_ui_click,
    "bass_drop": _gen_bass_drop, "glitch": _gen_glitch, "menu_select": _gen_menu_select,
    "heartbeat": _gen_heartbeat, "teleport": _gen_teleport, "chiptune_arp": _gen_chiptune_arp,
    "alarm_beep": _gen_alarm_beep,
    "glass_break": _gen_glass_break, "water_splash": _gen_water_splash,
    "wind": _gen_wind, "footsteps": _gen_footsteps, "portal": _gen_portal,
    "fanfare": _gen_fanfare, "error_buzz": _gen_error_buzz, "metal_clang": _gen_metal_clang,
    # v4 new
    "vinyl_crackle": _gen_vinyl_crackle, "thunder": _gen_thunder, "drone": _gen_drone,
    "chime": _gen_chime, "riser": _gen_riser, "sub_drop": _gen_sub_drop,
    "digital_noise": _gen_digital_noise, "crowd_cheer": _gen_crowd_cheer,
    "machinegun": _gen_machinegun, "alarm_clock": _gen_alarm_clock,
}

# Built-in presets
BUILTIN_PRESETS = [
    {"name": "Sci-Fi Blaster", "sfx_id": "laser", "params": {"freq_start": 1800, "freq_end": 120, "duration": 250, "volume": 85}, "fx": {"reverb": 20, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "distortion": 0, "lowpass": 20000, "highpass": 0, "pitch_semitones": 0, "reverse": False}},
    {"name": "Deep Explosion", "sfx_id": "explosion", "params": {"duration": 1500, "low_freq": 50, "noise_mix": 70, "decay": 1200, "volume": 90}, "fx": {"reverb": 50, "lowpass": 4000, "highpass": 0, "distortion": 0.8, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "pitch_semitones": -3, "reverse": False}},
    {"name": "Retro Coin", "sfx_id": "coin", "params": {"freq1": 1047, "freq2": 1319, "note_dur": 60, "volume": 75}, "fx": {"reverb": 10, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "distortion": 0, "lowpass": 8000, "highpass": 0, "pitch_semitones": 0, "reverse": False}},
    {"name": "Dramatic Power Up", "sfx_id": "powerup", "params": {"freq_start": 100, "freq_end": 2000, "duration": 1200, "steps": 10, "volume": 80}, "fx": {"reverb": 40, "delay_ms": 80, "delay_feedback": 30, "delay_mix": 20, "distortion": 0, "lowpass": 20000, "highpass": 0, "pitch_semitones": 0, "reverse": False}},
    {"name": "Cinematic Hit", "sfx_id": "hit", "params": {"duration": 400, "noise_mix": 60, "pitch": 80, "volume": 95}, "fx": {"reverb": 60, "lowpass": 5000, "distortion": 1.0, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "highpass": 0, "pitch_semitones": -5, "reverse": False}},
    {"name": "Ethereal Portal", "sfx_id": "portal", "params": {"duration": 2000, "res_freq": 180, "wobble": 70, "volume": 75}, "fx": {"reverb": 70, "delay_ms": 200, "delay_feedback": 40, "delay_mix": 30, "distortion": 0, "lowpass": 6000, "highpass": 0, "pitch_semitones": 0, "reverse": False, "chorus_rate": 1.5, "chorus_depth": 10, "chorus_mix": 0.4}},
    {"name": "Glitchy Error", "sfx_id": "error_buzz", "params": {"duration": 500, "frequency": 180, "pulse_rate": 8, "roughness": 60, "volume": 75}, "fx": {"distortion": 2.0, "lowpass": 5000, "reverb": 0, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "highpass": 0, "pitch_semitones": 0, "reverse": False}},
    {"name": "Epic Fanfare", "sfx_id": "fanfare", "params": {"duration": 2000, "base_freq": 220, "brilliance": 70, "volume": 80}, "fx": {"reverb": 60, "delay_ms": 150, "delay_feedback": 25, "delay_mix": 20, "distortion": 0, "lowpass": 20000, "highpass": 0, "pitch_semitones": 0, "reverse": False}},
    {"name": "Spooky Wind", "sfx_id": "wind", "params": {"duration": 5000, "gust_speed": 0.4, "hp_freq": 150, "volume": 65}, "fx": {"reverb": 30, "lowpass": 3000, "highpass": 0, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "distortion": 0, "pitch_semitones": 0, "reverse": False}},
    {"name": "Metal Impact", "sfx_id": "metal_clang", "params": {"duration": 1500, "freq": 1200, "harmonics": 8, "decay": 800, "volume": 85}, "fx": {"reverb": 45, "lowpass": 12000, "highpass": 0, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "distortion": 0, "pitch_semitones": 0, "reverse": False}},
    {"name": "Lo-Fi Vinyl", "sfx_id": "vinyl_crackle", "params": {"duration": 5000, "density": 50, "tone_freq": 3000, "volume": 55}, "fx": {"lowpass": 8000, "highpass": 80, "reverb": 0, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "distortion": 0, "pitch_semitones": 0, "reverse": False}},
    {"name": "Epic Thunder", "sfx_id": "thunder", "params": {"duration": 6000, "rumble_freq": 45, "crack_intensity": 80, "decay": 4000, "volume": 90}, "fx": {"reverb": 60, "lowpass": 5000, "highpass": 0, "delay_ms": 200, "delay_feedback": 30, "delay_mix": 20, "distortion": 0.5, "pitch_semitones": -2, "reverse": False}},
    {"name": "Dark Drone", "sfx_id": "drone", "params": {"duration": 8000, "base_freq": 55, "detune": 15, "harmonics": 6, "volume": 65}, "fx": {"reverb": 70, "lowpass": 4000, "highpass": 0, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "distortion": 0, "pitch_semitones": 0, "reverse": False, "chorus_rate": 0.3, "chorus_depth": 15, "chorus_mix": 0.4}},
    {"name": "Crystal Chime", "sfx_id": "chime", "params": {"duration": 3000, "fundamental": 1047, "brightness": 70, "attack": 3, "volume": 75}, "fx": {"reverb": 80, "lowpass": 15000, "highpass": 0, "delay_ms": 300, "delay_feedback": 40, "delay_mix": 35, "distortion": 0, "pitch_semitones": 0, "reverse": False}},
    {"name": "Tension Riser", "sfx_id": "riser", "params": {"duration": 5000, "freq_start": 80, "freq_end": 8000, "noise_mix": 50, "volume": 80}, "fx": {"reverb": 30, "lowpass": 20000, "highpass": 0, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "distortion": 0.5, "pitch_semitones": 0, "reverse": False}},
    {"name": "Earthquake Sub", "sfx_id": "sub_drop", "params": {"duration": 3000, "freq_start": 300, "freq_end": 20, "distortion": 1.0, "volume": 95}, "fx": {"reverb": 40, "lowpass": 500, "highpass": 0, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "distortion": 0, "pitch_semitones": -5, "reverse": False}},
    {"name": "Retro Glitch", "sfx_id": "digital_noise", "params": {"duration": 1500, "bit_depth": 3, "sr_div": 12, "volume": 70}, "fx": {"reverb": 0, "lowpass": 10000, "highpass": 200, "delay_ms": 80, "delay_feedback": 50, "delay_mix": 30, "distortion": 1.5, "pitch_semitones": 0, "reverse": False}},
    {"name": "Victory Crowd", "sfx_id": "crowd_cheer", "params": {"duration": 4000, "size": 80, "excitement": 85, "volume": 75}, "fx": {"reverb": 50, "lowpass": 10000, "highpass": 0, "delay_ms": 0, "delay_feedback": 0, "delay_mix": 0, "distortion": 0, "pitch_semitones": 0, "reverse": False}},
    {"name": "Warfare", "sfx_id": "machinegun", "params": {"duration": 3000, "fire_rate": 12, "tone": 1200, "echo_mix": 40, "volume": 80}, "fx": {"reverb": 40, "lowpass": 6000, "highpass": 0, "delay_ms": 100, "delay_feedback": 30, "delay_mix": 25, "distortion": 0.3, "pitch_semitones": 0, "reverse": False}},
]


def generate_sfx(sfx_id, params, fx_config=None, seed=None):
    """Generate a sound effect. Returns (samples_float32, display_name)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    vol = params.get("volume", 80) / 100.0
    func = _GENERATORS.get(sfx_id)
    if func is None:
        raise ValueError(f"Unknown SFX: {sfx_id}")
    samples = func(params, rng)
    samples = _to_mono_float(samples) * vol
    if fx_config:
        samples = apply_fx_chain(samples, fx_config)
    name = next((d["name"] for d in SFX_DEFS if d["id"] == sfx_id), sfx_id)
    return samples, name


def generate_layered(layers, seed=None):
    """Layer multiple SFX. Each layer: {"sfx_id","params","delay_ms","volume","fx_config"}"""
    parts, names = [], []
    for layer in layers:
        sid = layer["sfx_id"]
        params = layer.get("params", get_sfx_defaults(sid))
        fx = layer.get("fx_config")
        vol = layer.get("volume", 100) / 100.0
        samples, name = generate_sfx(sid, params, fx, seed)
        samples = samples * vol
        delay = int(layer.get("delay_ms", 0) / 1000.0 * SAMPLE_RATE)
        if delay > 0:
            padded = np.zeros(delay + len(samples), dtype=np.float32)
            padded[delay:] = samples; samples = padded
        parts.append(samples); names.append(name)
    ml = max(len(p) for p in parts)
    result = np.zeros(ml, dtype=np.float64)
    for p in parts: result[:len(p)] += p[:ml]
    peak = np.max(np.abs(result))
    if peak > 1e-6: result /= peak
    return result.astype(np.float32), " + ".join(names)


def get_sfx_defaults(sfx_id):
    for d in SFX_DEFS:
        if d["id"] == sfx_id:
            return {p[1]: p[2] for p in d["params"]}
    return {}

def get_categories():
    return sorted({d.get("category", "Other") for d in SFX_DEFS})


def export_audio(samples, path, sr=SAMPLE_RATE, format="WAV", subtype="PCM_16", channels=1, pan=0.0):
    """Export with format options."""
    if channels == 2:
        data = _to_stereo(samples, pan)
    else:
        data = samples
    sf.write(path, data, sr, format=format, subtype=subtype)


def export_wav(samples, path, sr=SAMPLE_RATE, subtype="PCM_16"):
    sf.write(path, samples, sr, subtype=subtype)


def samples_to_wav_bytes(samples, sr=SAMPLE_RATE, subtype="PCM_16"):
    buf = io.BytesIO()
    sf.write(buf, samples, sr, subtype=subtype, format="WAV")
    buf.seek(0); return buf.read()


def play_samples(samples, sr=SAMPLE_RATE):
    tmp = "/tmp/_sfx_preview.wav"; export_wav(samples, tmp, sr); _play_file(tmp)


def _play_file(path):
    try:
        import simpleaudio; wf = wave.open(path, "rb")
        sa_obj = simpleaudio.WaveObject(wf.readframes(wf.getnframes()), wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        wf.close(); sa_obj.play(); return
    except Exception: pass
    try:
        from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PySide6.QtCore import QUrl
        _p = QMediaPlayer(); _a = QAudioOutput(); _p.setAudioOutput(_a)
        _p.setSource(QUrl.fromLocalFile(os.path.abspath(path))); _a.setVolume(1.0); _p.play(); return
    except Exception: pass
    import subprocess
    try: subprocess.Popen(["aplay", "-q", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError: raise RuntimeError("No audio backend. Install simpleaudio.")


def save_preset(path, preset_name, sfx_id, params, fx_config=None):
    with open(path, "w") as f: json.dump({"preset_name": preset_name, "sfx_id": sfx_id, "params": params, "fx_config": fx_config or {}}, f, indent=2)

def load_preset(path):
    with open(path, "r") as f: return json.load(f)


def compute_fft(samples, sr=SAMPLE_RATE, n_bins=256):
    n = len(samples); windowed = samples * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed)); freqs = np.fft.rfftfreq(n, 1.0 / sr)
    if len(spectrum) > n_bins:
        idx = np.linspace(0, len(spectrum) - 1, n_bins).astype(int)
        return freqs[idx], spectrum[idx]
    return freqs, spectrum


# -- Undo Stack ---------------------------------------------------------------

class UndoStack:
    def __init__(self, limit=100):
        self._states = []
        self._index = -1
        self._limit = limit

    def push(self, state):
        self._states = self._states[:self._index + 1]
        self._states.append(state)
        if len(self._states) > self._limit:
            self._states.pop(0)
        self._index = len(self._states) - 1

    def undo(self):
        if self._index > 0:
            self._index -= 1; return self._states[self._index]
        return None

    def redo(self):
        if self._index < len(self._states) - 1:
            self._index += 1; return self._states[self._index]
        return None

    def can_undo(self): return self._index > 0
    def can_redo(self): return self._index < len(self._states) - 1


# -- Waveform Widget with Zoom, Selection & Playhead -------------------------

class WaveformWidget(QWidget):
    selectionChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = None
        self._zoom = 1.0
        self._scroll = 0.0
        self._sel_start = -1.0
        self._sel_end = -1.0
        self._dragging = False
        self._playhead = -1.0  # 0..1 fraction for animated playback
        self.setMinimumHeight(100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def set_samples(self, samples):
        self._samples = samples
        self._zoom = 1.0; self._scroll = 0.0
        self._sel_start = self._sel_end = -1.0
        self._playhead = -1.0
        self.update()

    def set_playhead(self, frac):
        self._playhead = frac
        self.update()

    def set_selection(self, start, end):
        self._sel_start = start; self._sel_end = end; self.update()

    def get_selection_samples(self):
        if self._samples is None or self._sel_start < 0 or self._sel_end < 0:
            return None
        s = min(self._sel_start, self._sel_end)
        e = max(self._sel_start, self._sel_end)
        n = len(self._samples)
        return self._samples[int(s * n):int(e * n)]

    def wheelEvent(self, event):
        if self._samples is None: return
        delta = event.angleDelta().y()
        factor = 1.2 if delta > 0 else 1 / 1.2
        self._zoom = min(max(self._zoom * factor, 1.0), 50.0)
        mx = event.position().x() / self.width()
        total = max(self._zoom - 1, 0)
        self._scroll = np.clip(self._scroll - (factor - 1) * mx / max(total, 0.001) * 0.3, 0, total)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._samples is not None:
            self._dragging = True
            frac = self._x_to_frac(event.position().x())
            self._sel_start = frac; self._sel_end = frac
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging and self._samples is not None:
            self._sel_end = self._x_to_frac(event.position().x())
            self.update()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            if self._sel_start >= 0 and self._sel_end >= 0:
                self.selectionChanged.emit(min(self._sel_start, self._sel_end),
                                           max(self._sel_start, self._sel_end))

    def _x_to_frac(self, x):
        w = self.width(); margin = 10; dw = w - 2 * margin
        if dw <= 0: return 0.0
        vis_start = self._scroll / max(self._zoom, 1)
        vis_end = vis_start + 1.0 / self._zoom
        return np.clip(vis_start + (x - margin) / dw * (vis_end - vis_start), 0, 1)

    def _frac_to_x(self, frac):
        w = self.width(); margin = 10; dw = w - 2 * margin
        vis_start = self._scroll / max(self._zoom, 1)
        vis_end = vis_start + 1.0 / self._zoom
        if vis_end <= vis_start: return margin
        return margin + (frac - vis_start) / (vis_end - vis_start) * dw

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0, QColor("#1a1a2e")); bg.setColorAt(1, QColor("#16213e"))
        painter.fillRect(self.rect(), bg)
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        for f in [0.25, 0.5, 0.75]: painter.drawLine(0, int(h*f), w, int(h*f))
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.DotLine))
        painter.drawLine(0, h//2, w, h//2)

        if self._samples is None or len(self._samples) == 0:
            painter.setPen(QColor(255,255,255,80)); painter.setFont(QFont("Segoe UI",11))
            painter.drawText(self.rect(), Qt.AlignCenter, "Generate a sound to see the waveform")
            painter.end(); return

        n = len(self._samples); margin = 10; dw = w - 2*margin; mid = h/2.0; amp = h/2.0 - 10
        vis_start = self._scroll / max(self._zoom, 1)
        vis_end = vis_start + 1.0 / self._zoom
        vis_start = np.clip(vis_start, 0, 1); vis_end = np.clip(vis_end, 0, 1)
        i_start = int(vis_start * n); i_end = int(vis_end * n)
        i_end = max(i_end, i_start + 1)
        vis_samples = self._samples[i_start:i_end]
        vis_n = len(vis_samples)
        max_pts = dw * 2
        if vis_n > max_pts:
            idx = np.linspace(0, vis_n - 1, max_pts).astype(int)
        else:
            idx = np.arange(vis_n)
        xs = margin + (idx / max(vis_n - 1, 1)) * dw
        ys = mid - vis_samples[idx] * amp

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor("#00f5d4")); grad.setColorAt(0.5, QColor("#7b2ff7")); grad.setColorAt(1, QColor("#f72585"))
        painter.setPen(QPen(QBrush(grad), 1.5))
        for i in range(len(xs) - 1):
            painter.drawLine(int(xs[i]), int(ys[i]), int(xs[i+1]), int(ys[i+1]))

        fg = QLinearGradient(0, 0, 0, h)
        fg.setColorAt(0, QColor(123,47,247,40)); fg.setColorAt(0.5, QColor(123,47,247,10)); fg.setColorAt(1, QColor(247,37,133,40))
        painter.setPen(Qt.NoPen); painter.setBrush(fg)
        path = QPainterPath(); path.moveTo(xs[0], mid)
        for x, y in zip(xs, ys): path.lineTo(x, y)
        for x, y in zip(reversed(xs), reversed(2*mid - ys)): path.lineTo(x, y)
        path.closeSubpath(); painter.drawPath(path)

        # Selection highlight
        if self._sel_start >= 0 and self._sel_end >= 0:
            s = min(self._sel_start, self._sel_end); e = max(self._sel_start, self._sel_end)
            x1 = int(self._frac_to_x(s)); x2 = int(self._frac_to_x(e))
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 245, 212, 30))
            painter.drawRect(x1, 0, x2 - x1, h)
            painter.setPen(QPen(QColor("#00f5d4", 150), 1))
            painter.drawLine(x1, 0, x1, h); painter.drawLine(x2, 0, x2, h)

        # Playhead
        if 0 <= self._playhead <= 1:
            ph_x = int(self._frac_to_x(self._playhead))
            painter.setPen(QPen(QColor("#ff4444", 200), 2))
            painter.drawLine(ph_x, 0, ph_x, h)
            # Playhead triangle
            painter.setBrush(QColor("#ff4444", 200))
            painter.setPen(Qt.NoPen)
            triangle = QPainterPath()
            triangle.moveTo(ph_x - 5, 0); triangle.lineTo(ph_x + 5, 0); triangle.lineTo(ph_x, 8)
            triangle.closeSubpath(); painter.drawPath(triangle)

        # Zoom indicator
        if self._zoom > 1.01:
            painter.setPen(QColor(255,255,255,120)); painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(5, 12, f"{self._zoom:.1f}x")

        # Minimap
        mm_h = 12; mm_y = h - mm_h
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 80))
        painter.drawRect(margin, mm_y, dw, mm_h)
        mm_pts = min(dw, n); mm_idx = np.linspace(0, n-1, mm_pts).astype(int)
        mm_xs = margin + (mm_idx / max(n-1, 1)) * dw
        mm_ys = mm_y + mm_h/2 - self._samples[mm_idx] * (mm_h/2 - 1)
        painter.setPen(QPen(QColor("#7b2ff7", 150), 1))
        for i in range(len(mm_xs) - 1):
            painter.drawLine(int(mm_xs[i]), int(np.clip(mm_ys[i], mm_y, mm_y+mm_h)),
                            int(mm_xs[i+1]), int(np.clip(mm_ys[i+1], mm_y, mm_y+mm_h)))
        vx1 = int(margin + vis_start * dw); vx2 = int(margin + vis_end * dw)
        painter.setPen(QPen(QColor("#00f5d4", 180), 1)); painter.setBrush(Qt.NoBrush)
        painter.drawRect(vx1, mm_y, vx2 - vx1, mm_h)
        # Playhead on minimap
        if 0 <= self._playhead <= 1:
            pmx = int(margin + self._playhead * dw)
            painter.setPen(QPen(QColor("#ff4444", 200), 1))
            painter.drawLine(pmx, mm_y, pmx, mm_y + mm_h)

        painter.end()


# -- Spectrum Widget ----------------------------------------------------------

class SpectrumWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._freqs = self._mags = None
        self.setMinimumHeight(80); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_spectrum(self, freqs, mags):
        self._freqs, self._mags = freqs, mags; self.update()

    def clear(self):
        self._freqs = self._mags = None; self.update()

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bg = QLinearGradient(0,0,0,h); bg.setColorAt(0,"#12122a"); bg.setColorAt(1,"#0e0e1e")
        painter.fillRect(self.rect(), bg)
        painter.setPen(QPen(QColor(255,255,255,20),1))
        for f in [0.25,0.5,0.75]: painter.drawLine(0,int(h*f),w,int(h*f))
        if self._freqs is None or len(self._mags) == 0:
            painter.setPen(QColor(255,255,255,80)); painter.setFont(QFont("Segoe UI",11))
            painter.drawText(self.rect(), Qt.AlignCenter, "Frequency spectrum"); painter.end(); return
        n = len(self._mags); margin = 10; dw = w - 2*margin; dh = h - 2*margin
        mm = np.max(self._mags) if np.max(self._mags) > 0 else 1.0; nm = self._mags / mm
        bw = max(dw / n - 1, 1)
        for i in range(n):
            bh = nm[i] * dh; x = margin + i * (dw / n); y = h - margin - bh
            ratio = i / max(n-1,1)
            if ratio < 0.33: c = QColor(0, int(245*ratio/0.33), int(212*ratio/0.33))
            elif ratio < 0.66:
                r2=(ratio-0.33)/0.33; c=QColor(int(123*r2),int(245-200*r2),int(212-165*r2))
            else:
                r3=(ratio-0.66)/0.34; c=QColor(int(123+124*r3),int(47-10*r3),int(47+100*r3))
            g = QLinearGradient(x,y,x,h-margin); g.setColorAt(0,c); g.setColorAt(1,QColor(c.red()//3,c.green()//3,c.blue()//3))
            painter.setPen(Qt.NoPen); painter.setBrush(g)
            painter.drawRoundedRect(int(x),int(y),max(int(bw),1),int(bh),1,1)
        painter.setPen(QColor(255,255,255,100)); painter.setFont(QFont("Segoe UI",7))
        for fl in [100,1000,5000,20000]:
            if self._freqs[-1] > 0:
                idx=int(np.searchsorted(self._freqs,fl))
                if 0<idx<n: painter.drawText(int(margin+idx*(dw/n))-10,h-1,str(fl))
        painter.end()


# -- Spectrogram (Waterfall) Widget -------------------------------------------

class SpectrogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._spectrogram_data = None
        self._n_bins = 128
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_samples(self, samples, n_bins=128, sr=SAMPLE_RATE):
        if samples is None or len(samples) < 256:
            self._spectrogram_data = None; self._qimage = None; self.update(); return
        n = len(samples)
        win_size = min(1024, n)
        hop = win_size // 4
        n_frames = max(1, (n - win_size) // hop)
        window = np.hanning(win_size)
        # Choose strategy: batch for large signals, loop for small
        if n_frames * win_size < 1_000_000:
            data = np.zeros((n_frames, n_bins))
            freqs = np.fft.rfftfreq(win_size, 1.0 / sr)
            bin_idx = np.linspace(0, len(freqs) - 1, n_bins).astype(int) if len(freqs) >= n_bins else None
            for i in range(n_frames):
                spectrum = np.abs(np.fft.rfft(samples[i * hop:i * hop + win_size] * window))
                if bin_idx is not None:
                    data[i] = spectrum[bin_idx]
                else:
                    data[i, :len(spectrum)] = spectrum
            self._freq_axis = freqs[bin_idx] if bin_idx is not None else np.linspace(0, sr/2, n_bins)
        else:
            indices = np.arange(win_size)[None, :] + np.arange(n_frames)[:, None] * hop
            indices = np.clip(indices, 0, n - 1)
            frames = samples[indices] * window
            spectra = np.abs(np.fft.rfft(frames, axis=1))
            freqs = np.fft.rfftfreq(win_size, 1.0 / sr)
            if len(freqs) > n_bins:
                bin_idx = np.linspace(0, len(freqs) - 1, n_bins).astype(int)
                data = spectra[:, bin_idx]
                self._freq_axis = freqs[bin_idx]
            else:
                data = np.zeros((n_frames, n_bins))
                data[:, :len(freqs)] = spectra
                self._freq_axis = np.linspace(0, sr/2, n_bins)
        data_db = np.clip(20 * np.log10(data + 1e-10), -80, 0)
        self._spectrogram_data = data_db
        self._time_axis = np.arange(n_frames) * hop / sr
        self._precompute_image()
        self.update()

    def _precompute_image(self):
        """Pre-render spectrogram to QImage for instant repaints."""
        if self._spectrogram_data is None:
            self._qimage = None; return
        n_frames, n_bins = self._spectrogram_data.shape
        norm = np.clip((self._spectrogram_data + 80) / 80.0, 0, 1)
        # Colormap: blue->cyan->green->yellow->red
        r_ch = np.clip(norm * 3 - 1, 0, 1)
        g_ch = np.clip(np.where(norm < 0.5, norm * 2, 2 - norm * 2), 0, 1)
        b_ch = np.clip(1 - norm * 2, 0, 1)
        # Build RGB array (flip vertically: low freq at bottom)
        img = np.zeros((n_bins, n_frames, 3), dtype=np.uint8)
        img[:, :, 0] = (r_ch * 255).astype(np.uint8)[::-1]
        img[:, :, 1] = (g_ch * 255).astype(np.uint8)[::-1]
        img[:, :, 2] = (b_ch * 255).astype(np.uint8)[::-1]
        from PySide6.QtGui import QImage
        self._qimage = QImage(img.data, n_frames, n_bins, n_frames * 3, QImage.Format_RGB888).copy()

    def clear(self):
        self._spectrogram_data = None; self._qimage = None; self.update()

    def paintEvent(self, event):
        painter = QPainter(self); w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#0e0e1e"))
        if self._spectrogram_data is None or self._qimage is None:
            painter.setPen(QColor(255,255,255,80)); painter.setFont(QFont("Segoe UI",11))
            painter.drawText(self.rect(), Qt.AlignCenter, "Spectrogram (waterfall)"); painter.end(); return
        n_frames, n_bins = self._spectrogram_data.shape
        margin_l, margin_r, margin_t, margin_b = 35, 5, 5, 18
        pw = w - margin_l - margin_r; ph = h - margin_t - margin_b
        if pw > 0 and ph > 0:
            painter.drawImage(margin_l, margin_t, self._qimage.scaled(pw, ph, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        painter.setPen(QColor(255,255,255,120)); painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(2, margin_t + 10, f"{self._freq_axis[-1]/1000:.0f}kHz")
        painter.drawText(2, h - margin_b - 2, f"{self._freq_axis[0]:.0f}Hz")
        painter.drawText(margin_l, h - 2, "0s")
        painter.drawText(w - margin_r - 30, h - 2, f"{self._time_axis[-1]:.1f}s")
        painter.end()


# -- VU Meter Widget ---------------------------------------------------------

class VUMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0; self._peak = 0.0
        self.setFixedSize(200, 16); self._timer = QTimer()
        self._timer.timeout.connect(self._decay); self._timer.start(50)

    def set_level(self, v):
        self._level = v; self._peak = max(self._peak, v); self.update()

    def _decay(self):
        self._peak = max(0, self._peak - 0.03); self.update()

    def paintEvent(self, event):
        painter = QPainter(self); w, h = self.width(), self.height()
        painter.setPen(Qt.NoPen); painter.setBrush(QColor("#1a1a2e"))
        painter.drawRoundedRect(0, 0, w, h, 3, 3)
        bar_w = int(self._level * (w - 4))
        if bar_w > 0:
            g = QLinearGradient(0, 0, w, 0)
            g.setColorAt(0, "#00f5d4"); g.setColorAt(0.7, "#7b2ff7"); g.setColorAt(1.0, "#f72585")
            painter.setBrush(g); painter.drawRoundedRect(2, 2, bar_w, h - 4, 2, 2)
        px = int(self._peak * (w - 4)) + 2
        if px < w - 2:
            painter.setBrush(QColor("#ffffff")); painter.drawRect(px, 1, 2, h - 2)
        painter.setPen(QColor(255,255,255,150)); painter.setFont(QFont("Segoe UI", 7))
        db = 20 * np.log10(max(self._level, 1e-6))
        painter.drawText(4, h - 3, f"{db:.0f}dB")
        painter.end()


# -- Param Slider ------------------------------------------------------------

class ParamSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label, min_val, max_val, default_val, parent=None):
        super().__init__(parent)
        self._label = label; self._min = min_val; self._max = max_val; self._default = default_val
        layout = QHBoxLayout(self); layout.setContentsMargins(0, 3, 0, 3)
        self.lbl = QLabel(label)
        self.lbl.setFixedWidth(160)
        self.lbl.setStyleSheet("color:#c0c0e0;font-size:12px;background:transparent;")
        layout.addWidget(self.lbl)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val)); self.slider.setMaximum(int(max_val))
        self.slider.setValue(int(default_val)); self.slider.setFixedHeight(22)
        layout.addWidget(self.slider, 1)
        self.val_label = QLabel(f"{int(default_val)}")
        self.val_label.setFixedWidth(50); self.val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val_label.setStyleSheet("color:#00f5d4;font-weight:bold;font-size:12px;background:transparent;")
        layout.addWidget(self.val_label)
        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, v):
        self.val_label.setText(str(v)); self.valueChanged.emit(float(v))

    def value(self): return float(self.slider.value())
    def setValue(self, v):
        self.slider.blockSignals(True); self.slider.setValue(int(v))
        self.val_label.setText(str(int(v))); self.slider.blockSignals(False)
    def reset(self): self.setValue(self._default)
    def randomize(self):
        self.setValue(float(np.random.randint(int(self._min), int(self._max)+1)))

    def contextMenuEvent(self, event):
        menu = QMenu(self); menu.setStyleSheet("QMenu{background:#1e1e3a;color:#e0e0e0;border:1px solid #3a3a5a;}")
        a_reset = menu.addAction("Reset to Default"); a_rand = menu.addAction("Randomize")
        act = menu.exec(event.globalPos())
        if act == a_reset: self.reset()
        elif act == a_rand: self.randomize()


# -- Stylesheet --------------------------------------------------------------

STYLE_SHEET = """
QMainWindow{background-color:#0f0f1a}
QLabel{color:#e0e0e0;font-size:13px;background:transparent}
QGroupBox{color:#b0b0d0;font-size:13px;font-weight:bold;border:1px solid #2a2a4a;border-radius:8px;
  margin-top:10px;padding:12px 8px 8px 8px;background-color:#15152a}
QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 5px}
QComboBox{background-color:#1e1e3a;color:#e0e0e0;border:1px solid #3a3a5a;border-radius:6px;
  padding:6px 10px;font-size:13px;min-height:18px}
QComboBox::drop-down{border:none;width:28px}
QComboBox::down-arrow{image:none;border-left:5px solid transparent;border-right:5px solid transparent;
  border-top:6px solid #7b2ff7;margin-right:8px}
QComboBox QAbstractItemView{background-color:#1e1e3a;color:#e0e0e0;border:1px solid #3a3a5a;
  selection-background-color:#7b2ff7;outline:none}
QSlider::groove:horizontal{height:5px;background:#2a2a4a;border-radius:2px}
QSlider::handle:horizontal{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #00f5d4,stop:1 #7b2ff7);
  width:14px;margin:-5px 0;border-radius:7px}
QSlider::sub-page:horizontal{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7b2ff7,stop:1 #00f5d4);border-radius:2px}
QPushButton{background-color:#7b2ff7;color:white;border:none;border-radius:6px;padding:8px 16px;
  font-size:13px;font-weight:bold}
QPushButton:hover{background-color:#9b4ff7}QPushButton:pressed{background-color:#5b1fd7}
QPushButton:disabled{background-color:#2a2a4a;color:#555}
QPushButton#playBtn{background-color:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #00c9a7,stop:1 #00f5d4);color:#0f0f1a;font-size:14px}
QPushButton#playBtn:hover{background-color:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #00f5d4,stop:1 #60ffda)}
QPushButton#exportBtn{background-color:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #f72585,stop:1 #ff6b9d)}
QPushButton#exportBtn:hover{background-color:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #ff6b9d,stop:1 #ff9dbf)}
QPushButton#stopBtn{background-color:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #ff4444,stop:1 #ff7777)}
QStatusBar{color:#888;background-color:#0a0a15;font-size:12px}
QScrollArea{border:none;background-color:transparent}
QTabWidget::pane{border:1px solid #2a2a4a;border-radius:6px;background:#15152a}
QTabBar::tab{background-color:#1a1a30;color:#8888aa;padding:8px 14px;border-top-left-radius:6px;
  border-top-right-radius:6px;margin-right:2px;font-size:12px}
QTabBar::tab:selected{background-color:#15152a;color:#e0e0e0}
QTabBar::tab:hover{color:#c0c0e0}
QListWidget{background-color:#12122a;color:#c0c0e0;border:1px solid #2a2a4a;border-radius:6px;font-size:12px;outline:none}
QListWidget::item{padding:5px;border-bottom:1px solid #1e1e3a}
QListWidget::item:selected{background-color:#7b2ff7}
QListWidget::item:hover{background-color:#2a2a4a}
QCheckBox{color:#c0c0e0;font-size:12px;background:transparent;spacing:6px}
QCheckBox::indicator{width:16px;height:16px;border-radius:4px;border:1px solid #3a3a5a;background-color:#1e1e3a}
QCheckBox::indicator:checked{background-color:#7b2ff7;border-color:#9b4ff7}
QSpinBox,QDoubleSpinBox{background-color:#1e1e3a;color:#e0e0e0;border:1px solid #3a3a5a;border-radius:4px;
  padding:3px 6px;font-size:12px}
QToolBar{background-color:#0f0f1a;border:none;spacing:5px;padding:3px}
QToolButton{background-color:#1e1e3a;color:#c0c0e0;border:1px solid #2a2a4a;border-radius:6px;padding:5px 8px;font-size:11px}
QToolButton:hover{background-color:#2a2a4a}
QToolButton:checked{background-color:#7b2ff7;color:white}
QLineEdit{background-color:#12122a;color:#e0e0e0;border:1px solid #3a3a4a;border-radius:4px;padding:4px 8px;font-size:12px}
QMenu{background-color:#1e1e3a;color:#e0e0e0;border:1px solid #3a3a5a}
QMenu::item:selected{background-color:#7b2ff7}
QToolTip{background-color:#1e1e3a;color:#e0e0e0;border:1px solid #3a3a5a;padding:4px;font-size:12px}
QProgressBar{background:#2a2a4a;border-radius:3px;border:none;text-align:center;color:#e0e0e0;font-size:10px}
QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #00f5d4,stop:1 #7b2ff7);border-radius:3px}
"""


# -- Layer Row ---------------------------------------------------------------

class LayerRow(QWidget):
    removeRequested = Signal(int)
    moveUpRequested = Signal(int)
    moveDownRequested = Signal(int)

    def __init__(self, index, parent=None):
        super().__init__(parent); self.index = index
        layout = QHBoxLayout(self); layout.setContentsMargins(2,2,2,2); layout.setSpacing(3)
        # Move buttons
        up = QPushButton("^"); up.setFixedSize(20,20)
        up.setStyleSheet("QPushButton{background:#2a2a4a;color:#aaa;border-radius:3px;padding:0;font-size:11px;font-weight:bold}")
        up.clicked.connect(lambda: self.moveUpRequested.emit(self.index))
        dn = QPushButton("v"); dn.setFixedSize(20,20)
        dn.setStyleSheet("QPushButton{background:#2a2a4a;color:#aaa;border-radius:3px;padding:0;font-size:11px;font-weight:bold}")
        dn.clicked.connect(lambda: self.moveDownRequested.emit(self.index))
        layout.addWidget(up); layout.addWidget(dn)
        self.combo = QComboBox()
        for d in SFX_DEFS: self.combo.addItem(d["name"], d["id"])
        self.combo.setFixedWidth(120); self.combo.setFixedHeight(26)
        layout.addWidget(self.combo)
        layout.addWidget(QLabel("Delay:"))
        self.delay_spin = QSpinBox(); self.delay_spin.setRange(0,5000); self.delay_spin.setValue(0)
        self.delay_spin.setFixedWidth(55); self.delay_spin.setSuffix("ms"); layout.addWidget(self.delay_spin)
        layout.addWidget(QLabel("Vol:"))
        self.vol_slider = QSlider(Qt.Horizontal); self.vol_slider.setRange(0,100); self.vol_slider.setValue(100)
        self.vol_slider.setFixedWidth(70); layout.addWidget(self.vol_slider)
        self.vol_label = QLabel("100%"); self.vol_label.setFixedWidth(32)
        self.vol_label.setStyleSheet("color:#00f5d4;font-size:11px;background:transparent;")
        self.vol_slider.valueChanged.connect(lambda v: self.vol_label.setText(f"{v}%"))
        layout.addWidget(self.vol_label)
        # Per-layer FX toggle
        self.fx_btn = QPushButton("FX"); self.fx_btn.setFixedSize(30,20)
        self.fx_btn.setCheckable(True)
        self.fx_btn.setStyleSheet("QPushButton{background:#2a2a4a;color:#888;border-radius:3px;padding:0;font-size:10px}"
                                  "QPushButton:checked{background:#7b2ff7;color:white}")
        self.fx_btn.setToolTip("Toggle per-layer FX")
        layout.addWidget(self.fx_btn)
        # Crossfade
        layout.addWidget(QLabel("Xfade:"))
        self.xfade_spin = QSpinBox(); self.xfade_spin.setRange(0,500); self.xfade_spin.setValue(50)
        self.xfade_spin.setSuffix("ms"); self.xfade_spin.setFixedWidth(55)
        self.xfade_spin.setToolTip("Crossfade duration between this and next layer")
        layout.addWidget(self.xfade_spin)
        rm = QPushButton("X"); rm.setFixedSize(24,24)
        rm.setStyleSheet("QPushButton{background:#ff4444;border-radius:4px;padding:0;font-weight:bold;font-size:13px}")
        rm.clicked.connect(lambda: self.removeRequested.emit(self.index)); layout.addWidget(rm)

    def get_layer(self):
        return {"sfx_id": self.combo.currentData(), "params": get_sfx_defaults(self.combo.currentData()),
                "delay_ms": self.delay_spin.value(), "volume": self.vol_slider.value(),
                "xfade_ms": self.xfade_spin.value(), "use_fx": self.fx_btn.isChecked()}


# -- Main Window -------------------------------------------------------------

class SFXGeneratorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SFX Generator Pro v4")
        self.setMinimumSize(1200, 800); self.resize(1350, 880)
        self.setStyleSheet(STYLE_SHEET)
        self._current_samples = None; self._current_name = ""
        self._param_widgets: list[ParamSlider] = []; self._fx_widgets: dict[str, ParamSlider] = {}
        self._history: list[dict] = []; self._layer_rows: list[LayerRow] = []
        self._auto_preview = True; self._loop_count = 1; self._seed = None
        self._favorites: set = set()
        self._undo_stack = UndoStack(); self._ab_samples = None; self._ab_name = ""
        self._layer_samples = None; self._layer_name = ""
        self._debounce = QTimer(); self._debounce.setSingleShot(True); self._debounce.setInterval(80)
        self._debounce.timeout.connect(self._generate)
        # Playback animation
        self._playback_timer = QTimer(); self._playback_timer.setInterval(16)  # ~60fps
        self._playback_timer.timeout.connect(self._animate_playhead)
        self._playback_start_time = 0; self._playback_duration = 0
        self._build_ui(); self._setup_shortcuts()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self._play)
        QShortcut(QKeySequence("G"), self, self._generate)
        QShortcut(QKeySequence("R"), self, self._randomize_params)
        QShortcut(QKeySequence("F"), self, self._toggle_favorite)
        QShortcut(QKeySequence("B"), self, self._toggle_ab)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self._undo)
        QShortcut(QKeySequence("Shift+Backspace"), self, self._redo)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_preset)
        QShortcut(QKeySequence("Ctrl+E"), self, self._export)
        QShortcut(QKeySequence("Ctrl+R"), self, self._reset_params)
        QShortcut(QKeySequence("1"), self, lambda: self.tabs.setCurrentIndex(0))
        QShortcut(QKeySequence("2"), self, lambda: self.tabs.setCurrentIndex(1))
        QShortcut(QKeySequence("3"), self, lambda: self.tabs.setCurrentIndex(2))

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        ml = QVBoxLayout(central); ml.setSpacing(5); ml.setContentsMargins(8,8,8,8)
        # Toolbar
        tb = QToolBar(); tb.setMovable(False); self.addToolBar(tb)
        tb.addWidget(QLabel("  Category:"))
        self.cat_combo = QComboBox(); self.cat_combo.addItem("All","all")
        self.cat_combo.addItem("Favorites","favorites")
        for c in get_categories(): self.cat_combo.addItem(c,c)
        self.cat_combo.setFixedWidth(110); self.cat_combo.currentIndexChanged.connect(self._on_category_changed)
        tb.addWidget(self.cat_combo)

        tb.addWidget(QLabel(" Search:"))
        self.search_edit = QLineEdit(); self.search_edit.setFixedWidth(140)
        self.search_edit.setPlaceholderText("Filter sounds...")
        self.search_edit.textChanged.connect(self._on_search_changed)
        tb.addWidget(self.search_edit)

        tb.addSeparator()
        self.auto_btn = QToolButton(); self.auto_btn.setCheckable(True); self.auto_btn.setChecked(True)
        self.auto_btn.setText("Auto-Preview"); self.auto_btn.setToolTip("Regenerate on slider change")
        self.auto_btn.toggled.connect(lambda c: setattr(self, '_auto_preview', c)); tb.addWidget(self.auto_btn)
        self.rt_btn = QToolButton(); self.rt_btn.setCheckable(True); self.rt_btn.setChecked(False)
        self.rt_btn.setText("Real-time"); self.rt_btn.setToolTip("Continuous update while dragging")
        tb.addWidget(self.rt_btn)

        tb.addSeparator()
        tb.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(-1, 999999); self.seed_spin.setValue(-1)
        self.seed_spin.setFixedWidth(70); self.seed_spin.setToolTip("-1 = random, 0+ = reproducible")
        self.seed_spin.setSpecialValueText("Random"); tb.addWidget(self.seed_spin)

        tb.addSeparator()
        sb = QToolButton(); sb.setText("Save Preset"); sb.clicked.connect(self._save_preset); tb.addWidget(sb)
        lb = QToolButton(); lb.setText("Load Preset"); lb.clicked.connect(self._load_preset); tb.addWidget(lb)
        self.preset_combo = QComboBox(); self.preset_combo.setFixedWidth(140)
        self.preset_combo.addItem("-- Built-in Presets --", -1)
        for i, p in enumerate(BUILTIN_PRESETS): self.preset_combo.addItem(p["name"], i)
        self.preset_combo.currentIndexChanged.connect(self._on_builtin_preset)
        tb.addWidget(self.preset_combo)

        tb.addSeparator()
        bb = QToolButton(); bb.setText("Batch Export All"); bb.clicked.connect(self._batch_export); tb.addWidget(bb)

        # Tabs
        self.tabs = QTabWidget(); ml.addWidget(self.tabs, 1)
        self._build_single_tab(); self._build_layer_tab(); self._build_spectrogram_tab()
        n_sfx = len(SFX_DEFS)
        self.statusBar().showMessage(
            f"Ready  |  {n_sfx} SFX  |  15 FX  |  Spectrogram  |  Playback Anim  |  Normalization  |  "
            f"Space=Play  G=Generate  R=Random  B=A/B  Ctrl+S=Save  Ctrl+E=Export")

    def _build_single_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab); layout.setSpacing(6); layout.setContentsMargins(4,4,4,4)
        # Left panel
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFixedWidth(365)
        lc = QWidget(); ll = QVBoxLayout(lc); ll.setContentsMargins(2,2,2,2); ll.setSpacing(3)
        # SFX selector
        sr = QHBoxLayout()
        sl = QLabel("Sound:"); sl.setFixedWidth(48)
        sl.setStyleSheet("color:#b0b0d0;font-weight:bold;background:transparent;"); sr.addWidget(sl)
        self.sfx_combo = QComboBox(); self._populate_combo("all"); self.sfx_combo.setFixedHeight(32)
        sr.addWidget(self.sfx_combo, 1)
        self.fav_btn = QPushButton("*"); self.fav_btn.setFixedSize(28,28)
        self.fav_btn.setStyleSheet("QPushButton{background:#2a2a4a;color:#666;border-radius:4px;font-size:16px;font-weight:bold}")
        self.fav_btn.setToolTip("Toggle favorite (F)"); self.fav_btn.clicked.connect(self._toggle_favorite)
        sr.addWidget(self.fav_btn)
        ll.addLayout(sr)
        # Buttons
        br = QHBoxLayout()
        rb = QPushButton("Randomize"); rb.clicked.connect(self._randomize_params); br.addWidget(rb)
        rsb = QPushButton("Reset"); rsb.clicked.connect(self._reset_params); br.addWidget(rsb)
        ub = QPushButton("Undo"); ub.clicked.connect(self._undo); br.addWidget(ub)
        redo_btn = QPushButton("Redo"); redo_btn.clicked.connect(self._redo); br.addWidget(redo_btn)
        ll.addLayout(br)
        # Params
        self.params_group = QGroupBox("SFX Parameters")
        self.params_grid = QVBoxLayout(self.params_group)
        self.params_grid.setSpacing(1); self.params_grid.setContentsMargins(4,14,4,4)
        ll.addWidget(self.params_group)
        # FX
        self.fx_group = QGroupBox("Post-Processing FX")
        fxl = QVBoxLayout(self.fx_group); fxl.setSpacing(1); fxl.setContentsMargins(4,14,4,4)
        fx_params = [
            ("Reverb (%)", "reverb", 0, 0, 100), ("Delay Time (ms)", "delay_ms", 0, 0, 1000),
            ("Delay Feedback (%)", "delay_feedback", 0, 0, 90), ("Delay Mix (%)", "delay_mix", 0, 0, 80),
            ("Distortion", "distortion", 0, 0, 500), ("Low-pass (Hz)", "lowpass", 20000, 100, 20000),
            ("High-pass (Hz)", "highpass", 0, 0, 5000), ("Pitch (semi)", "pitch_semitones", 0, -24, 24),
            ("Chorus Rate (Hz)", "chorus_rate", 0, 0, 10), ("Chorus Depth (ms)", "chorus_depth", 7, 1, 20),
            ("Chorus Mix (%)", "chorus_mix", 0, 0, 80), ("Tremolo Rate (Hz)", "tremolo_rate", 0, 0, 20),
            ("Compressor Thresh (dB)", "compressor_threshold", -60, -60, 0),
            ("EQ Low (dB)", "eq_low", 0, -12, 12), ("EQ Mid (dB)", "eq_mid", 0, -12, 12),
            ("EQ High (dB)", "eq_high", 0, -12, 12),
            # v4 new FX
            ("Flanger Rate (Hz)", "flanger_rate", 0, 0, 5), ("Flanger Depth (ms)", "flanger_depth", 3, 1, 15),
            ("Flanger Feedback (%)", "flanger_feedback", 50, 0, 90), ("Flanger Mix (%)", "flanger_mix", 50, 0, 80),
            ("Phaser Rate (Hz)", "phaser_rate", 0, 0, 5), ("Phaser Depth", "phaser_depth", 50, 0, 100),
            ("Phaser Feedback (%)", "phaser_feedback", 30, 0, 80), ("Phaser Mix (%)", "phaser_mix", 50, 0, 80),
            ("Bitcrusher Bits", "bitcrusher_bits", 0, 0, 16), ("Bitcrusher SR Div", "bitcrusher_sr", 100, 2, 100),
        ]
        for label, key, default, min_v, max_v in fx_params:
            s = ParamSlider(label, min_v, max_v, default); fxl.addWidget(s); self._fx_widgets[key] = s
        self.reverse_cb = QCheckBox("Reverse"); self.reverse_cb.setStyleSheet("margin-top:3px;margin-left:4px;")
        fxl.addWidget(self.reverse_cb)
        # Normalization
        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("Normalize:"))
        self.norm_combo = QComboBox(); self.norm_combo.setFixedWidth(80)
        self.norm_combo.addItem("Off", "off"); self.norm_combo.addItem("Peak", "peak")
        self.norm_combo.addItem("RMS", "rms"); self.norm_combo.addItem("LUFS", "lufs")
        norm_row.addWidget(self.norm_combo)
        norm_row.addWidget(QLabel("Target:"))
        self.norm_target = QDoubleSpinBox(); self.norm_target.setRange(-60, 0)
        self.norm_target.setValue(-3.0); self.norm_target.setSingleStep(0.5)
        self.norm_target.setSuffix(" dB"); self.norm_target.setFixedWidth(70)
        norm_row.addWidget(self.norm_target)
        norm_row.addStretch()
        fxl.addLayout(norm_row)
        rfx = QPushButton("Reset FX"); rfx.setFixedHeight(26); rfx.clicked.connect(self._reset_fx); fxl.addWidget(rfx)
        ll.addWidget(self.fx_group); ll.addStretch()
        left_scroll.setWidget(lc); layout.addWidget(left_scroll)
        # Right panel
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(5)
        # A/B indicator + VU meter
        ab_row = QHBoxLayout()
        self.ab_label = QLabel(""); self.ab_label.setStyleSheet("color:#00f5d4;font-size:12px;font-weight:bold;background:transparent;")
        ab_row.addWidget(self.ab_label); ab_row.addStretch()
        self.vu_meter = VUMeter(); ab_row.addWidget(self.vu_meter)
        rl.addLayout(ab_row)
        # Visualizations
        vs = QSplitter(Qt.Vertical)
        self.waveform = WaveformWidget(); vs.addWidget(self.waveform)
        self.spectrum = SpectrumWidget(); vs.addWidget(self.spectrum)
        vs.setSizes([300, 180]); rl.addWidget(vs, 1)
        # Control buttons
        cr = QHBoxLayout()
        gb = QPushButton("Generate"); gb.clicked.connect(self._generate); cr.addWidget(gb)
        self.play_btn = QPushButton("Play"); self.play_btn.setObjectName("playBtn"); self.play_btn.clicked.connect(self._play); self.play_btn.setEnabled(False); cr.addWidget(self.play_btn)
        stop_btn = QPushButton("Stop"); stop_btn.setObjectName("stopBtn"); stop_btn.clicked.connect(self._stop_playback); cr.addWidget(stop_btn)
        self.ab_btn = QPushButton("A/B"); self.ab_btn.setToolTip("Compare current with stored snapshot (B)")
        self.ab_btn.clicked.connect(self._toggle_ab); cr.addWidget(self.ab_btn)
        cr.addWidget(QLabel("Loop:"))
        self.loop_spin = QSpinBox(); self.loop_spin.setRange(1,50); self.loop_spin.setValue(1); self.loop_spin.setFixedWidth(50)
        self.loop_spin.valueChanged.connect(lambda v: setattr(self,'_loop_count',v)); cr.addWidget(self.loop_spin)
        cr.addStretch()
        # Export format
        self.fmt_combo = QComboBox(); self.fmt_combo.addItem("WAV 16-bit", ("WAV","PCM_16"))
        self.fmt_combo.addItem("WAV 24-bit", ("WAV","PCM_24")); self.fmt_combo.addItem("WAV 32-bit float", ("WAV","FLOAT"))
        self.fmt_combo.addItem("FLAC", ("FLAC","PCM_16")); self.fmt_combo.addItem("OGG", ("OGG","VORBIS"))
        self.fmt_combo.setFixedWidth(120); cr.addWidget(self.fmt_combo)
        self.export_btn = QPushButton("Export"); self.export_btn.setObjectName("exportBtn"); self.export_btn.clicked.connect(self._export); self.export_btn.setEnabled(False); cr.addWidget(self.export_btn)
        rl.addLayout(cr)
        # Selection controls
        sel_row = QHBoxLayout()
        self.psb = QPushButton("Play Selection"); self.psb.clicked.connect(self._play_selection); self.psb.setEnabled(False)
        sel_row.addWidget(self.psb)
        self.esb = QPushButton("Export Selection"); self.esb.clicked.connect(self._export_selection); self.esb.setEnabled(False)
        sel_row.addWidget(self.esb)
        csb = QPushButton("Clear Selection"); csb.clicked.connect(self._clear_selection); sel_row.addWidget(csb)
        self.waveform.selectionChanged.connect(lambda s,e: (self.psb.setEnabled(True), self.esb.setEnabled(True)))
        rl.addLayout(sel_row)
        self.info_label = QLabel(""); self.info_label.setStyleSheet("color:#888;font-size:11px;background:transparent;")
        self.info_label.setAlignment(Qt.AlignCenter); rl.addWidget(self.info_label)
        # History
        hr = QHBoxLayout(); hl = QLabel("History:"); hl.setStyleSheet("color:#666;font-size:11px;background:transparent;")
        hr.addWidget(hl)
        self.history_list = QListWidget(); self.history_list.setFixedHeight(55); self.history_list.setSpacing(0)
        self.history_list.currentRowChanged.connect(self._on_history_selected); hr.addWidget(self.history_list, 1)
        rl.addLayout(hr)
        layout.addWidget(right, 1)
        self.tabs.addTab(tab, "Single SFX")
        self.sfx_combo.currentIndexChanged.connect(self._on_sfx_type_changed)
        self._on_sfx_type_changed(0)

    def _build_spectrogram_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); layout.setContentsMargins(8,8,8,8); layout.setSpacing(5)
        info = QLabel("Spectrogram view of the currently generated sound (Single SFX tab)")
        info.setStyleSheet("color:#888;font-size:12px;background:transparent;"); layout.addWidget(info)
        self.spectrogram = SpectrogramWidget(); layout.addWidget(self.spectrogram, 1)
        # Controls
        ctrl = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Spectrogram"); refresh_btn.clicked.connect(self._update_spectrogram)
        ctrl.addWidget(refresh_btn); ctrl.addStretch()
        layout.addLayout(ctrl)
        self.tabs.addTab(tab, "Spectrogram")

    def _build_layer_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); layout.setSpacing(6); layout.setContentsMargins(8,8,8,8)
        info = QLabel("Layer multiple SFX with volume, delay, crossfade, and per-layer FX")
        info.setStyleSheet("color:#888;font-size:12px;background:transparent;"); layout.addWidget(info)
        self.layers_container = QWidget(); self.layers_layout = QVBoxLayout(self.layers_container)
        self.layers_layout.setContentsMargins(0,0,0,0); self.layers_layout.setSpacing(4)
        ls = QScrollArea(); ls.setWidgetResizable(True); ls.setWidget(self.layers_container)
        layout.addWidget(ls, 1)
        lbr = QHBoxLayout()
        alb = QPushButton("+ Add Layer"); alb.clicked.connect(self._add_layer); lbr.addWidget(alb); lbr.addStretch()
        self.layer_gen_btn = QPushButton("Generate Layered"); self.layer_gen_btn.setObjectName("playBtn")
        self.layer_gen_btn.clicked.connect(self._generate_layered); lbr.addWidget(self.layer_gen_btn)
        self.layer_play_btn = QPushButton("Play"); self.layer_play_btn.clicked.connect(self._play_layered)
        self.layer_play_btn.setEnabled(False); lbr.addWidget(self.layer_play_btn)
        self.layer_export_btn = QPushButton("Export"); self.layer_export_btn.setObjectName("exportBtn")
        self.layer_export_btn.clicked.connect(self._export_layered); self.layer_export_btn.setEnabled(False); lbr.addWidget(self.layer_export_btn)
        layout.addLayout(lbr)
        self.layer_waveform = WaveformWidget(); self.layer_waveform.setMinimumHeight(100); layout.addWidget(self.layer_waveform)
        self.layer_info = QLabel(""); self.layer_info.setStyleSheet("color:#888;font-size:11px;background:transparent;")
        self.layer_info.setAlignment(Qt.AlignCenter); layout.addWidget(self.layer_info)
        self._add_layer(); self._add_layer()
        self.tabs.addTab(tab, "Layer Mixer")

    # -- Combo -----------------------------------------------------------
    def _populate_combo(self, category, search=""):
        self.sfx_combo.blockSignals(True)
        cid = self.sfx_combo.currentData() if self.sfx_combo.count() > 0 else None
        self.sfx_combo.clear()
        search = search.lower()
        for d in SFX_DEFS:
            if category == "favorites" and d["id"] not in self._favorites: continue
            if category != "all" and category != "favorites" and d.get("category","Other") != category: continue
            if search and search not in d["name"].lower() and search not in d["id"].lower(): continue
            star = " * " if d["id"] in self._favorites else ""
            self.sfx_combo.addItem(star + d["name"], d["id"])
        if cid:
            idx = self.sfx_combo.findData(cid)
            if idx >= 0: self.sfx_combo.setCurrentIndex(idx)
        self.sfx_combo.blockSignals(False)

    def _on_category_changed(self, _):
        self._populate_combo(self.cat_combo.currentData(), self.search_edit.text())
        if self.sfx_combo.count() > 0: self._on_sfx_type_changed(0)

    def _on_search_changed(self, text):
        self._populate_combo(self.cat_combo.currentData(), text)

    # -- SFX params ------------------------------------------------------
    def _on_sfx_type_changed(self, idx):
        if idx < 0: return
        for w in self._param_widgets: w.setParent(None); w.deleteLater()
        self._param_widgets.clear()
        while self.params_grid.count():
            item = self.params_grid.takeAt(0); w = item.widget()
            if w: w.setParent(None); w.deleteLater()
        sid = self.sfx_combo.currentData()
        sfx_def = next((d for d in SFX_DEFS if d["id"] == sid), None)
        if not sfx_def: return
        for param in sfx_def["params"]:
            label, key, default, min_v, max_v = param
            s = ParamSlider(label, min_v, max_v, default)
            if self._auto_preview: s.valueChanged.connect(self._on_param_live)
            self.params_grid.addWidget(s); self._param_widgets.append(s)
        self.params_grid.addStretch()
        self._update_fav_btn()
        QTimer.singleShot(50, self._generate)

    def _on_param_live(self, _=None):
        if self._auto_preview:
            if self.rt_btn.isChecked(): self._generate()
            else: self._debounce.start()

    def _get_params(self):
        sid = self.sfx_combo.currentData()
        sfx_def = next((d for d in SFX_DEFS if d["id"] == sid), None)
        if not sfx_def: return {}
        return {p[1]: self._param_widgets[i].value() for i, p in enumerate(sfx_def["params"]) if i < len(self._param_widgets)}

    def _get_fx_config(self):
        fx = {k: w.value() for k, w in self._fx_widgets.items()}
        fx["reverse"] = self.reverse_cb.isChecked()
        fx["normalize_mode"] = self.norm_combo.currentData()
        fx["normalize_target"] = self.norm_target.value()
        return fx

    def _has_active_fx(self):
        fx = self._get_fx_config()
        return any([fx.get("reverb",0)>0, fx.get("delay_ms",0)>0, fx.get("distortion",0)>0,
                    fx.get("lowpass",20000)<20000, fx.get("highpass",0)>0, fx.get("pitch_semitones",0)!=0,
                    fx.get("reverse",False), fx.get("chorus_rate",0)>0, fx.get("tremolo_rate",0)>0,
                    fx.get("compressor_threshold",-60)>-60, fx.get("eq_low",0)!=0, fx.get("eq_mid",0)!=0,
                    fx.get("eq_high",0)!=0, fx.get("flanger_rate",0)>0, fx.get("phaser_rate",0)>0,
                    fx.get("bitcrusher_bits",0)>0, fx.get("normalize_mode","off")!="off"])

    def _randomize_params(self):
        sid = self.sfx_combo.currentData()
        sfx_def = next((d for d in SFX_DEFS if d["id"] == sid), None)
        if sfx_def:
            for i, w in enumerate(self._param_widgets):
                _, _, _, min_v, max_v = sfx_def["params"][i]
                w.setValue(float(np.random.randint(int(min_v), int(max_v)+1)))
        self._generate()

    def _reset_params(self):
        sid = self.sfx_combo.currentData()
        sfx_def = next((d for d in SFX_DEFS if d["id"] == sid), None)
        if sfx_def:
            for i, w in enumerate(self._param_widgets): w.setValue(float(sfx_def["params"][i][2]))
        self._generate()

    def _reset_fx(self):
        defaults = {"reverb":0,"delay_ms":0,"delay_feedback":0,"delay_mix":0,"distortion":0,
                     "lowpass":20000,"highpass":0,"pitch_semitones":0,"chorus_rate":0,"chorus_depth":7,
                     "chorus_mix":0,"tremolo_rate":0,"compressor_threshold":-60,"eq_low":0,"eq_mid":0,"eq_high":0,
                     "flanger_rate":0,"flanger_depth":3,"flanger_feedback":50,"flanger_mix":50,
                     "phaser_rate":0,"phaser_depth":50,"phaser_feedback":30,"phaser_mix":50,
                     "bitcrusher_bits":0,"bitcrusher_sr":100}
        for k,v in defaults.items():
            if k in self._fx_widgets: self._fx_widgets[k].setValue(float(v))
        self.reverse_cb.setChecked(False); self.norm_combo.setCurrentIndex(0)
        self.norm_target.setValue(-3.0); self._generate()

    def _generate(self):
        sid = self.sfx_combo.currentData(); params = self._get_params()
        fx = self._get_fx_config() if self._has_active_fx() else None
        seed = self.seed_spin.value() if self.seed_spin.value() >= 0 else None
        t0 = time.time()
        try:
            samples, name = generate_sfx(sid, params, fx, seed)
            elapsed = (time.time()-t0)*1000
            self._current_samples = samples; self._current_name = name
            self._update_visuals(samples, name, elapsed)
            self._undo_stack.push({"sfx_id": sid, "params": dict(params), "fx": dict(fx) if fx else None})
            self._add_history(sid, name, params, fx, samples)
            self.play_btn.setEnabled(True); self.export_btn.setEnabled(True)
            peak = np.max(np.abs(samples))
            self.vu_meter.set_level(float(peak))
            # Update spectrogram if tab exists
            if hasattr(self, 'spectrogram') and self.spectrogram is not None:
                self.spectrogram.set_samples(samples)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Generation failed:\n{e}")

    def _update_spectrogram(self):
        if self._current_samples is not None:
            self.spectrogram.set_samples(self._current_samples)
            self.statusBar().showMessage("Spectrogram updated")

    def _update_visuals(self, samples, name, elapsed=0):
        self.waveform.set_samples(samples)
        freqs, mags = compute_fft(samples, n_bins=128)
        self.spectrum.set_spectrum(freqs, mags)
        dur = len(samples)/SAMPLE_RATE*1000
        fx_txt = " + FX" if self._has_active_fx() else ""
        norm_mode = self._get_fx_config().get("normalize_mode", "off")
        norm_txt = f" [{norm_mode.upper()}]" if norm_mode != "off" else ""
        self.info_label.setText(f"{name}{fx_txt}{norm_txt}  |  {dur:.0f}ms  |  {SAMPLE_RATE}Hz  |  {len(samples):,} samples  |  {elapsed:.0f}ms gen")
        self.statusBar().showMessage(f"Generated: {name} in {elapsed:.0f}ms")

    # -- Favorites -------------------------------------------------------
    def _toggle_favorite(self):
        sid = self.sfx_combo.currentData()
        if sid in self._favorites: self._favorites.discard(sid)
        else: self._favorites.add(sid)
        self._update_fav_btn()
        if self.cat_combo.currentData() == "favorites":
            self._populate_combo("favorites", self.search_edit.text())

    def _update_fav_btn(self):
        sid = self.sfx_combo.currentData()
        is_fav = sid in self._favorites
        self.fav_btn.setText("*" if is_fav else "-")
        color = "#ffcc00" if is_fav else "#666"
        self.fav_btn.setStyleSheet(f"QPushButton{{background:#2a2a4a;color:{color};border-radius:4px;font-size:16px;font-weight:bold}}")

    # -- A/B Compare ----------------------------------------------------
    def _toggle_ab(self):
        if self._ab_samples is None:
            self._ab_samples = self._current_samples
            self._ab_name = self._current_name
            self.ab_label.setText(f"A/B: Stored '{self._ab_name}' as B")
            self.statusBar().showMessage(f"A/B: Snapshot stored. Press B again to compare.")
        else:
            self._ab_samples, self._current_samples = self._current_samples, self._ab_samples
            self._ab_name, self._current_name = self._current_name, self._ab_name
            self._update_visuals(self._current_samples, self._current_name)
            self.ab_label.setText(f"A/B: Showing '{self._current_name}'  (stored: '{self._ab_name}')")
            self.statusBar().showMessage(f"A/B: Now showing '{self._current_name}'")

    # -- Undo/Redo ------------------------------------------------------
    def _undo(self):
        state = self._undo_stack.undo()
        if state:
            idx = self.sfx_combo.findData(state["sfx_id"])
            if idx >= 0: self.sfx_combo.setCurrentIndex(idx)
            QTimer.singleShot(100, lambda: self._apply_state(state))

    def _redo(self):
        state = self._undo_stack.redo()
        if state:
            idx = self.sfx_combo.findData(state["sfx_id"])
            if idx >= 0: self.sfx_combo.setCurrentIndex(idx)
            QTimer.singleShot(100, lambda: self._apply_state(state))

    def _apply_state(self, state):
        sid = state["sfx_id"]
        sfx_def = next((d for d in SFX_DEFS if d["id"] == sid), None)
        if sfx_def:
            params = state.get("params", {})
            for i, w in enumerate(self._param_widgets):
                if i < len(sfx_def["params"]):
                    key = sfx_def["params"][i][1]
                    if key in params: w.setValue(float(params[key]))
        fx = state.get("fx", {}) or {}
        for k, v in fx.items():
            if k == "reverse": self.reverse_cb.setChecked(v)
            elif k == "normalize_mode":
                idx = self.norm_combo.findData(v)
                if idx >= 0: self.norm_combo.setCurrentIndex(idx)
            elif k == "normalize_target": self.norm_target.setValue(float(v))
            elif k in self._fx_widgets: self._fx_widgets[k].setValue(float(v))
        self._generate()

    # -- Playback with Animation ----------------------------------------
    def _play(self):
        if self._current_samples is not None:
            try:
                s = self._current_samples if self._loop_count <= 1 else np.tile(self._current_samples, self._loop_count)
                play_samples(s); self.vu_meter.set_level(1.0)
                self.statusBar().showMessage(f"Playing: {self._current_name}")
                # Start playback animation
                self._playback_duration = len(s) / SAMPLE_RATE
                self._playback_start_time = time.time()
                self._playback_timer.start()
            except Exception as e: QMessageBox.warning(self, "Error", str(e))

    def _stop_playback(self):
        self._playback_timer.stop()
        self.waveform.set_playhead(-1.0)

    def _animate_playhead(self):
        elapsed = time.time() - self._playback_start_time
        frac = elapsed / self._playback_duration if self._playback_duration > 0 else 1.0
        if frac >= 1.0:
            self._playback_timer.stop()
            self.waveform.set_playhead(-1.0)
        else:
            self.waveform.set_playhead(frac)

    def _play_selection(self):
        sel = self.waveform.get_selection_samples()
        if sel is not None and len(sel) > 0:
            try: play_samples(sel); self.statusBar().showMessage("Playing selection")
            except Exception as e: QMessageBox.warning(self, "Error", str(e))

    def _export_selection(self):
        sel = self.waveform.get_selection_samples()
        if sel is None or len(sel) == 0: return
        path, _ = QFileDialog.getSaveFileName(self, "Export Selection", "selection.wav", "WAV Files (*.wav)")
        if path:
            try: export_wav(sel, path); self.statusBar().showMessage(f"Exported: {path}")
            except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def _clear_selection(self):
        self.waveform.set_selection(-1, -1)

    # -- Export ----------------------------------------------------------
    def _export(self):
        if self._current_samples is None: return
        fmt, subtype = self.fmt_combo.currentData()
        ext = fmt.lower()
        default = self._current_name.replace(" ","_").replace("/","_").lower() + f".{ext}"
        path, _ = QFileDialog.getSaveFileName(self, "Export Audio", default,
                    f"{fmt} Files (*.{ext});;All Files (*)")
        if path:
            try:
                export_audio(self._current_samples, path, format=fmt, subtype=subtype)
                self.statusBar().showMessage(f"Exported: {path}")
                QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
            except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def _batch_export(self):
        d = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not d: return
        count = 0; total = len(SFX_DEFS)
        progress = QProgressDialog("Batch exporting...", "Cancel", 0, total, self)
        progress.setWindowModality(Qt.WindowModal); progress.setMinimumDuration(0)
        try:
            for i, sfx_def in enumerate(SFX_DEFS):
                if progress.wasCanceled(): break
                progress.setValue(i)
                QApplication.processEvents()
                defaults = get_sfx_defaults(sfx_def["id"])
                samples, _ = generate_sfx(sfx_def["id"], defaults)
                export_wav(samples, os.path.join(d, sfx_def["id"] + ".wav")); count += 1
            progress.setValue(total)
            self.statusBar().showMessage(f"Batch exported {count} SFX to: {d}")
            QMessageBox.information(self, "Batch Export", f"Exported {count} sounds to:\n{d}")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    # -- Presets ---------------------------------------------------------
    def _on_builtin_preset(self, idx):
        if idx <= 0: return
        p = BUILTIN_PRESETS[idx - 1]
        ci = self.sfx_combo.findData(p["sfx_id"])
        if ci >= 0: self.sfx_combo.setCurrentIndex(ci)
        QTimer.singleShot(100, lambda: self._apply_state({
            "sfx_id": p["sfx_id"], "params": dict(p["params"]), "fx": dict(p["fx"])
        }))

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip(): return
        path, _ = QFileDialog.getSaveFileName(self, "Save", f"{name.strip()}.json", "JSON (*.json)")
        if path:
            try: save_preset(path, name.strip(), self.sfx_combo.currentData(), self._get_params(), self._get_fx_config())
            except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON (*.json)")
        if path:
            try:
                preset = load_preset(path); idx = self.sfx_combo.findData(preset["sfx_id"])
                if idx >= 0: self.sfx_combo.setCurrentIndex(idx)
                QTimer.singleShot(100, lambda: self._apply_state(preset))
            except Exception as e: QMessageBox.critical(self, "Error", str(e))

    # -- History ---------------------------------------------------------
    def _add_history(self, sid, name, params, fx, samples):
        self._history.append({"sfx_id": sid, "name": name, "params": dict(params),
                              "fx": dict(fx) if fx else None, "samples": samples.copy()})
        if len(self._history) > 50: self._history = self._history[-50:]
        self.history_list.blockSignals(True)
        self.history_list.addItem(f"[{len(self._history)}] {name}")
        self.history_list.setCurrentRow(self.history_list.count()-1)
        self.history_list.blockSignals(False); self.history_list.scrollToBottom()

    def _on_history_selected(self, row):
        if 0 <= row < len(self._history):
            e = self._history[row]; idx = self.sfx_combo.findData(e["sfx_id"])
            if idx >= 0: self.sfx_combo.setCurrentIndex(idx)
            QTimer.singleShot(100, lambda: self._apply_state(e))

    # -- Layer Mixer -----------------------------------------------------
    def _add_layer(self):
        i = len(self._layer_rows); r = LayerRow(i)
        r.removeRequested.connect(self._remove_layer)
        r.moveUpRequested.connect(self._move_layer_up)
        r.moveDownRequested.connect(self._move_layer_down)
        self._layer_rows.append(r); self.layers_layout.addWidget(r)

    def _remove_layer(self, idx):
        if len(self._layer_rows) <= 1: return
        r = self._layer_rows[idx]; r.setParent(None); r.deleteLater(); self._layer_rows.pop(idx)
        for i, r2 in enumerate(self._layer_rows): r2.index = i

    def _move_layer_up(self, idx):
        if idx <= 0: return
        self._layer_rows[idx-1], self._layer_rows[idx] = self._layer_rows[idx], self._layer_rows[idx-1]
        self._layer_rows[idx-1].index = idx-1; self._layer_rows[idx].index = idx
        # Reorder in layout
        self.layers_layout.removeWidget(self._layer_rows[idx-1])
        self.layers_layout.removeWidget(self._layer_rows[idx])
        self.layers_layout.insertWidget(idx-1, self._layer_rows[idx-1])
        self.layers_layout.insertWidget(idx, self._layer_rows[idx])

    def _move_layer_down(self, idx):
        if idx >= len(self._layer_rows) - 1: return
        self._move_layer_up(idx + 1)

    def _generate_layered(self):
        layers = [r.get_layer() for r in self._layer_rows]
        try:
            samples, name = generate_layered(layers)
            self._layer_samples = samples; self._layer_name = name
            self.layer_waveform.set_samples(samples)
            self.layer_play_btn.setEnabled(True); self.layer_export_btn.setEnabled(True)
            self.layer_info.setText(f"{name}  |  {len(samples)/SAMPLE_RATE*1000:.0f}ms  |  {len(layers)} layers  |  {len(samples):,} samples")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def _play_layered(self):
        if self._layer_samples is not None:
            try: play_samples(self._layer_samples)
            except Exception as e: QMessageBox.warning(self, "Error", str(e))

    def _export_layered(self):
        if self._layer_samples is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "layered_sfx.wav", "WAV (*.wav)")
        if path:
            try: export_wav(self._layer_samples, path); QMessageBox.information(self, "Done", f"Saved:\n{path}")
            except Exception as e: QMessageBox.critical(self, "Error", str(e))


def main():
    app = QApplication(sys.argv); app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window, QColor("#0f0f1a")); p.setColor(QPalette.WindowText, QColor("#e0e0e0"))
    p.setColor(QPalette.Base, QColor("#1a1a2e")); p.setColor(QPalette.AlternateBase, QColor("#15152a"))
    p.setColor(QPalette.Text, QColor("#e0e0e0")); p.setColor(QPalette.Button, QColor("#1e1e3a"))
    p.setColor(QPalette.ButtonText, QColor("#e0e0e0")); p.setColor(QPalette.Highlight, QColor("#7b2ff7"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff")); app.setPalette(p)
    w = SFXGeneratorWindow(); w.show(); sys.exit(app.exec())

if __name__ == "__main__": main()