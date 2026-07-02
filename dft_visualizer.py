#!/usr/bin/env python3
"""
DFT Audio Visualizer — Full Edition (Fixed Peak Annotations)
Features:
 - PyQt / pyqtgraph real-time visualizer
 - 2D spectrum + scrolling spectrogram
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
        if self.prev_spec is None:
            self.prev_spec = mag.copy()
            return False, 0.0
        diff = mag - self.prev_spec
        flux = np.sum(np.maximum(diff, 0.0))
        self.prev_spec = mag.copy()
        self.env = self.smooth * self.env + (1 - self.smooth) * flux
        # detect onset when flux > env * thr
        onset = flux > max(1e-12, self.env * self.thr)
        return onset, flux

# Peak detector: find top N local peaks in magnitude spectrum
def detect_peaks(mag, freqs, top_n=5, mindb=-120.0):
    mags_db = db_amp(mag, floor_db=mindb)
    peak_idx = signal.find_peaks(mags_db, distance=2)[0]
    if len(peak_idx) == 0:
        return []
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
        interval = max(10, int(self.hop / float(self.sr) * 1000.0))
        self.timer.setInterval(interval)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

    def _build_ui(self):
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
        self.window_combo.currentTextChanged.connect(self._on_window_change)
        ctrl_row.addWidget(QtWidgets.QLabel("Window:"))
        ctrl_row.addWidget(self.window_combo)

        # smoothing
        self.smooth_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(0, 99)
        self.smooth_slider.setValue(int(self.smooth_alpha*99))
        self.smooth_slider.valueChanged.connect(self._on_smooth_change)
        ctrl_row.addWidget(QtWidgets.QLabel("Smoothing:"))
        ctrl_row.addWidget(self.smooth_slider)

        # gain
        self.gain_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(1, 400)
        self.gain_slider.setValue(int(self.gain*100))
        self.gain_slider.valueChanged.connect(self._on_gain_change)
        ctrl_row.addWidget(QtWidgets.QLabel("Gain:"))
        ctrl_row.addWidget(self.gain_slider)

        # buttons row
        btn_row = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_row)
        
        self.btn_snapshot = QtWidgets.QPushButton("Snapshot")
        self.btn_snapshot.clicked.connect(self._on_snapshot)
        btn_row.addWidget(self.btn_snapshot)

        self.btn_record = QtWidgets.QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.toggled.connect(self._on_record_toggle)
        btn_row.addWidget(self.btn_record)

        self.btn_save_preset = QtWidgets.QPushButton("Save Preset")
        self.btn_save_preset.clicked.connect(self._on_save_preset)
        btn_row.addWidget(self.btn_save_preset)

        self.btn_load_preset = QtWidgets.QPushButton("Load Preset")
        self.btn_load_preset.clicked.connect(self._on_load_preset)
        btn_row.addWidget(self.btn_load_preset)

        # plotting area: spectrum (top) and spectrogram (bottom)
        plot_split = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(plot_split)

        # spectrum plot
        self.plot_widget = pg.PlotWidget(title="Spectrum")
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_widget.setLabel('left', 'Magnitude', units='dB')
        self.plot_widget.showGrid(True, True, alpha=0.3)
        self.spectrum_curve = self.plot_widget.plot([], pen='y')
        plot_split.addWidget(self.plot_widget)

        # annotate peaks via text items
        self.peak_text_items = []

        # spectrogram (image)
        self.img_view = ImageView()
        plot_split.addWidget(self.img_view)

    # -------------------
    # UI callbacks
    # -------------------
    def _on_fft_change(self, txt):
        try:
            n = int(txt)
            self.fft_size = n
            self._recalc_freqs()
        except Exception:
            pass

    def _on_hop_change(self, val):
        self.hop = int(val)
        interval = max(10, int(self.hop / float(self.sr) * 1000.0))
        self.timer.setInterval(interval)

    def _on_window_change(self, txt):
        self.window_name = txt
        self.window_fn = WINDOWS.get(txt, WINDOWS['hann'])

    def _on_smooth_change(self, val):
        self.smooth_alpha = val / 99.0

    def _on_gain_change(self, val):
        self.gain = val / 100.0

    def _on_snapshot(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"spectrogram_snapshot_{ts}.png"
        img = self.img_view.getImageItem().image
        exporter = pg.exporters.ImageExporter(self.img_view.getView())
        exporter.parameters()['width'] = img.shape[1]
        exporter.export(fname)
        print("Saved snapshot:", fname)

    def _on_record_toggle(self, toggled):
        if toggled:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"recording_{ts}.wav"
            self.recorder.start(fname)
            self.btn_record.setText("Stop")
        else:
            self.recorder.stop()
            self.btn_record.setText("Record")

    def _on_save_preset(self):
        preset = {
            'fft_size': self.fft_size,
            'hop': self.hop,
            'window': self.window_name,
            'smooth': self.smooth_alpha,
            'gain': self.gain,
            'spec_len': self.spec_len,
        }
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Preset", "", "JSON Files (*.json)")
        if fname:
            with open(fname, 'w') as f:
                json.dump(preset, f, indent=2)
            print("Preset saved:", fname)

    def _on_load_preset(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON Files (*.json)")
        if fname:
            with open(fname, 'r') as f:
                preset = json.load(f)
            self.fft_size = int(preset.get('fft_size', self.fft_size))
            self.hop = int(preset.get('hop', self.hop))
            self.window_name = preset.get('window', self.window_name)
            self.window_fn = WINDOWS.get(self.window_name, WINDOWS['hann'])
            self.smooth_alpha = float(preset.get('smooth', self.smooth_alpha))
            self.gain = float(preset.get('gain', self.gain))
            self.spec_len = int(preset.get('spec_len', self.spec_len))
            self.fft_combo.setCurrentText(str(self.fft_size))
            self.hop_spin.setValue(self.hop)
            self.window_combo.setCurrentText(self.window_name)
            self.smooth_slider.setValue(int(self.smooth_alpha*99))
            self.gain_slider.setValue(int(self.gain*100))
            print("Preset loaded:", fname)
            self._recalc_freqs()

    def _recalc_freqs(self):
        self.freqs = np.fft.rfftfreq(self.fft_size, 1.0/self.sr)
        self.sgram = np.full((self.freqs.size, self.spec_len), self.db_floor, dtype=float)
        self.smooth_spec = None
        self.onset_detector = OnsetDetector(self.fft_size, thr=self.onset_thr)

    # -------------------
    # Timer update
    # -------------------
    def _on_timer(self):
        block = self.audio_source.read_block()
        if block is None:
            return
        self.buffer = np.concatenate([self.buffer, block])
        if self.buffer.size < self.fft_size:
            return
        frame = self.buffer[:self.fft_size]
        self.buffer = self.buffer[self.hop:]

        # compute magnitude
        mag = compute_spectrum(frame * self.gain, self.fft_size, self.window_fn)
        onset, flux = self.onset_detector.feed(mag)
        self.recorder.push(frame)

        # smoothing in dB
        mag_db = db_amp(mag, floor_db=self.db_floor)
        if self.smooth_spec is None:
            self.smooth_spec = mag_db
        else:
            self.smooth_spec = self.smooth_alpha * self.smooth_spec + (1.0 - self.smooth_alpha) * mag_db

        # update spectrum curve
        self.spectrum_curve.setData(self.freqs, self.smooth_spec)
        self.plot_widget.setYRange(self.db_floor, DEFAULT_DB_CEILING)

        # detect peaks
        peaks = detect_peaks(mag, self.freqs, top_n=self.peak_count, mindb=self.db_floor)
        
        # clear previous texts
        for it in self.peak_text_items:
            self.plot_widget.removeItem(it)
        self.peak_text_items = []
        
        # Render peaks in log10 space
        for (f, dbv, idx) in peaks:
            if f <= 0:
                continue
            note, cents, _ = freq_to_note_name(f)
            txt = f"{f:.1f} Hz\n{note} {cents:+.1f}c"
            ti = pg.TextItem(txt, anchor=(0.5, 1.0), color='w')
            ti.setPos(math.log10(f), dbv)
            
            self.plot_widget.addItem(ti)
            self.peak_text_items.append(ti)

        # spectrogram update
        self.sgram = np.roll(self.sgram, -1, axis=1)
        self.sgram[:, -1] = mag_db
        self.img_view.setImage(self.sgram, autoLevels=False, autoRange=False)

        # Visual feedback for onsets
        if onset:
            self.plot_widget.setBackground('k')
            self.setWindowTitle("DFT Visualizer — ONSET!")
            QtCore.QTimer.singleShot(150, lambda: self.setWindowTitle("DFT Visualizer — Full Edition"))

# ---------------------------
# CLI and main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="DFT Audio Visualizer — Full Edition")
    parser.add_argument('--mode', choices=['live', 'file'], required=True)
    parser.add_argument('--file', type=str, help='If mode=file, path to audio file')
    parser.add_argument('--sr', type=int, default=DEFAULT_SR)
    parser.add_argument('--fft', type=int, default=DEFAULT_FFT)
    parser.add_argument('--hop', type=int, default=DEFAULT_HOP)
    parser.add_argument('--spec_len', type=int, default=DEFAULT_SPEC_LEN)
    args = parser.parse_args()

    if args.mode == 'file':
        if not args.file:
            print("Error: --file required when mode=file")
            sys.exit(1)
        src = FileAudioSource(args.file, blocksize=args.hop)
        sr = src.sr
    else:
        src = LiveAudioSource(sr=args.sr, blocksize=args.hop)
        sr = args.sr

    app = QtWidgets.QApplication(sys.argv)
    vis = VisualizerApp(src, sr, fft_size=args.fft, hop=args.hop, spec_len=args.spec_len)
    vis.resize(1000, 800)
    vis.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
