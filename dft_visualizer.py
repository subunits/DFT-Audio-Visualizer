#!/usr/bin/env python3
"""
DFT Audio Visualizer — Full Edition (Fixed Peak Annotations)
Features:
 - PyQt / pyqtgraph real-time visualizer
 - 2D spectrum + scrolling spectrogram
 - 3D waterfall (OpenGL) if available
 - Peak detection + nearest musical note names (A4=440Hz)
 - Onset detection (spectral flux) and beat-sync visual effects
 - GUI controls (FFT size, hop, window, smoothing, gain, colormap)
 - Presets save/load
 - Recording (write to WAV), snapshot export
 - Two audio sources: live mic (sounddevice) or file (soundfile)
Usage:
    python dft_visualizer_full.py --mode live
    python dft_visualizer_full.py --mode file --file path/to.wav
Dependencies:
    pip install numpy scipy sounddevice soundfile pyqtgraph PyQt6
"""

import argparse
import json
import math
import os
import sys
import time
import threading
import collections
from datetime import datetime

import numpy as np
from scipy import signal
import sounddevice as sd
import soundfile as sf

# GUI / plotting
try:
    from PyQt6 import QtCore, QtWidgets
    from PyQt6.QtCore import Qt
except Exception:
    from PyQt5 import QtCore, QtWidgets
    from PyQt5.QtCore import Qt

import pyqtgraph as pg
from pyqtgraph import ImageView

# Optional OpenGL waterfall
try:
    import pyqtgraph.opengl as gl
    GL_AVAILABLE = True
except Exception:
    GL_AVAILABLE = False

# ---------------------------
# Defaults / parameters
# ---------------------------
DEFAULT_SR = 44100
DEFAULT_FFT = 4096
DEFAULT_HOP = 1024
DEFAULT_SPEC_LEN = 300
DEFAULT_SMOOTH = 0.7
DEFAULT_DB_FLOOR = -100.0
DEFAULT_DB_CEILING = 0.0
WINDOWS = {
    'hann': np.hanning,
    'hamming': np.hamming,
    'blackman': np.blackman,
    'rect': lambda N: np.ones(N),
}

# ---------------------------
# Utilities: freq->note
# ---------------------------
A4_FREQ = 440.0
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note_name(freq):
    """Return (note_name, cents_offset, octave) nearest to freq."""
    if freq <= 0:
        return ("--", 0.0, None)
    # MIDI note number
    midi = 69 + 12 * math.log2(freq / A4_FREQ)
    midi_round = int(round(midi))
    cents = (midi - midi_round) * 100.0
    note_idx = midi_round % 12
    octave = (midi_round // 12) - 1
    return (f"{NOTE_NAMES[note_idx]}{octave}", cents, octave)

# ---------------------------
# DSP: FFT, window, dB
# ---------------------------
def db_amp(mag, floor_db=DEFAULT_DB_FLOOR):
    with np.errstate(divide='ignore'):
        db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    return np.clip(db, floor_db, DEFAULT_DB_CEILING)

def compute_spectrum(frame, fft_size, window_fn):
    if frame.shape[0] < fft_size:
        frame = np.pad(frame, (0, fft_size - frame.shape[0]), mode='constant')
    w = window_fn(fft_size)
    frame_win = frame[:fft_size] * w
    spec = np.fft.rfft(frame_win)
    mag = np.abs(spec) / np.sum(w)
    return mag

# Simple spectral flux onset detector
class OnsetDetector:
    def __init__(self, fft_size, thr=4.0, smooth=0.9):
        self.prev_spec = None
        self.thr = thr
        self.smooth = smooth
        self.env = 0.0
        self.onset_flag = False

    def feed(self, mag):
        # mag is linear magnitude (not dB)
        if self.prev_spec is None:
            self.prev_spec = mag.copy()
            return False, 0.0
        # spectral flux: half-wave of (mag - prev)
        diff = mag - self.prev_spec
        flux = np.sum(np.maximum(diff, 0.0))
        self.prev_spec = mag.copy()
        # simple adaptive threshold via envelope
        self.env = self.smooth * self.env + (1 - self.smooth) * flux
        # detect onset when flux > env * thr
        onset = flux > max(1e-12, self.env * self.thr)
        return onset, flux

# Peak detector: find top N local peaks in magnitude spectrum
def detect_peaks(mag, freqs, top_n=5, mindb=-120.0):
    # mag: linear magnitude
    # convert to dB for thresholding
    mags_db = db_amp(mag, floor_db=mindb)
    # find local maxima
    peak_idx = signal.find_peaks(mags_db, distance=2)[0]
    if len(peak_idx) == 0:
        return []
    # sort peaks by magnitude
    sorted_idx = peak_idx[np.argsort(mags_db[peak_idx])[::-1]]
    results = []
    for idx in sorted_idx[:top_n]:
        results.append((freqs[idx], mags_db[idx], idx))
    return results

# ---------------------------
# Audio sources
# ---------------------------
class FileAudioSource:
    def __init__(self, path, blocksize, mono=True):
        self.path = path
        self.blocksize = blocksize
        self.f, self.sr = sf.read(path, always_2d=True)
        if mono and self.f.shape[1] > 1:
            self.f = self.f.mean(axis=1, keepdims=True)
        self.pos = 0
        self.lock = threading.Lock()

    def read_block(self):
        with self.lock:
            if self.pos >= self.f.shape[0]:
                return None
            end = min(self.pos + self.blocksize, self.f.shape[0])
            block = self.f[self.pos:end, 0].copy()
            if block.size < self.blocksize:
                block = np.pad(block, (0, self.blocksize - block.size))
            self.pos = end
            return block

class LiveAudioSource:
    def __init__(self, sr=DEFAULT_SR, blocksize=DEFAULT_HOP, channels=1, device=None):
        self.sr = sr
        self.blocksize = blocksize
        self.channels = channels
        self.buffer = collections.deque()
        self.lock = threading.Lock()
        self.stream = sd.InputStream(samplerate=self.sr, blocksize=self.blocksize,
                                     channels=self.channels, callback=self._callback,
                                     device=device)
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status, file=sys.stderr)
        block = indata[:, 0].copy()
        with self.lock:
            self.buffer.append(block)
            # limit buffer length
            while len(self.buffer) > 200:
                self.buffer.popleft()

    def read_block(self):
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer.popleft()

# ---------------------------
# Recorder
# ---------------------------
class Recorder:
    def __init__(self, sr, channels=1, outpath=None):
        self.sr = sr
        self.channels = channels
        self.outpath = outpath
        self._buf = []
        self.recording = False
        self.lock = threading.Lock()

    def start(self, outpath=None):
        if outpath:
            self.outpath = outpath
        if not self.outpath:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.outpath = f"recording_{ts}.wav"
        with self.lock:
            self._buf = []
            self.recording = True
        print("Recording to:", self.outpath)

    def stop(self):
        with self.lock:
            self.recording = False
            buf = np.concatenate(self._buf) if self._buf else np.zeros((0,))
            self._buf = []
        if buf.size:
            sf.write(self.outpath, buf, self.sr)
            print("Saved recording:", self.outpath)
        else:
            print("No audio recorded.")

    def push(self, block):
        with self.lock:
            if self.recording:
                self._buf.append(block.copy())

# ---------------------------
# GUI / Visualizer (PyQt + pyqtgraph)
# ---------------------------
class VisualizerApp(QtWidgets.QMainWindow):
    def __init__(self, audio_source, sr, fft_size=DEFAULT_FFT, hop=DEFAULT_HOP,
                 spec_len=DEFAULT_SPEC_LEN):
        super().__init__()
        self.setWindowTitle("DFT Visualizer — Full Edition")
        self.audio_source = audio_source
        self.sr = sr
        self.fft_size = fft_size
        self.hop = hop
        self.spec_len = spec_len

        # internal buffer for overlapping frames
        self.buffer = np.zeros(0, dtype=np.float32)
        self.window_name = 'hann'
        self.window_fn = WINDOWS[self.window_name]
        self.smooth_alpha = DEFAULT_SMOOTH
        self.db_floor = DEFAULT_DB_FLOOR
        self.gain = 1.0
        self.peak_count = 6
        self.onset_thr = 4.0

        # spectral storage
        self.freqs = np.fft.rfftfreq(self.fft_size, 1.0/self.sr)
        self.sgram = np.full((self.freqs.size, self.spec_len), self.db_floor, dtype=float)
        self.smooth_spec = None
        self.onset_detector = OnsetDetector(self.fft_size, thr=self.onset_thr)

        # recorder
        self.recorder = Recorder(self.sr)

        # build UI
        self._build_ui()

        # timer for update
        self.timer = QtCore.QTimer()
        # aim for ~hop / sr * 1000 ms, ensure not too fast
        interval = max(10, int(self.hop / float(self.sr) * 1000.0))
        self.timer.setInterval(interval)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

        # For 3D waterfall history (GL)
        if GL_AVAILABLE:
            self.waterfall_mesh = None
            self.waterfall_history = collections.deque(maxlen=self.spec_len)

    def _build_ui(self):
        # central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout()
        central.setLayout(layout)

        # Top controls row
        ctrl_row = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl_row)

        # FFT size dropdown
        self.fft_combo = QtWidgets.QComboBox()
        for n in [512, 1024, 2048, 4096, 8192]:
            self.fft_combo.addItem(str(n))
        self.fft_combo.setCurrentText(str(self.fft_size))
        self.fft_combo.currentTextChanged.connect(self._on_fft_change)
        ctrl_row.addWidget(QtWidgets.QLabel("FFT size:"))
        ctrl_row.addWidget(self.fft_combo)

        # hop
        self.hop_spin = QtWidgets.QSpinBox()
        self.hop_spin.setRange(64, 8192)
        self.hop_spin.setValue(self.hop)
        self.hop_spin.valueChanged.connect(self._on_hop_change)
        ctrl_row.addWidget(QtWidgets.QLabel("Hop:"))
        ctrl_row.addWidget(self.hop_spin)

        # window
        self.window_combo = QtWidgets.QComboBox()
        for w in WINDOWS.keys():
            self.window_combo.addItem(w)
        self.window_combo.setCurrentText(self.window_name)
        self.window_combo.currentTextChanged.connect(self._on
