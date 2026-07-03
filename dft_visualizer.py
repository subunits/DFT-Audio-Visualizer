#!/usr/bin/env python3
"""
DFT Audio Visualizer — Full Edition (Hardened & Polished, v3.1-PATCHED)

CHANGES FROM v3.1:
 ✓ Fixed peak detection early-return bug (line 608: return → continue)
 ✓ Added window-close race condition protection (is_closing flag)
 ✓ Added bounds checking for peak_count
 ✓ Added exception handling in peak label cleanup
 ✓ Added onset threshold UI slider
 ✓ Improved error messages with status feedback
 ✓ Protected FFT size change with flag
"""

import argparse
import json
import math
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
    QT_HORIZONTAL = Qt.Orientation.Horizontal
    QT_VERTICAL = Qt.Orientation.Vertical
except Exception:
    from PyQt5 import QtCore, QtWidgets
    from PyQt5.QtCore import Qt
    QT_HORIZONTAL = Qt.Horizontal
    QT_VERTICAL = Qt.Vertical

import pyqtgraph as pg
import pyqtgraph.exporters

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
MIN_HOP = 64
FFT_CHOICES = [512, 1024, 2048, 4096, 8192]
COLORMAPS = ['viridis', 'plasma', 'inferno', 'grey']

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
    if freq <= 0:
        return ("--", 0.0, None)
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
    denom = np.sum(w)
    if denom <= 0:
        denom = 1.0
    mag = np.abs(spec) / denom
    return mag


class OnsetDetector:
    def __init__(self, thr=4.0, smooth=0.9, warmup_frames=8):
        self.prev_spec = None
        self.thr = thr
        self.smooth = smooth
        self.env = None
        self.warmup_frames = warmup_frames
        self._frame_count = 0

    def feed(self, mag):
        if self.prev_spec is None:
            self.prev_spec = mag.copy()
            return False, 0.0

        diff = mag - self.prev_spec
        flux = np.sum(np.maximum(diff, 0.0))
        self.prev_spec = mag.copy()

        if self.env is None:
            self.env = flux
        else:
            self.env = self.smooth * self.env + (1 - self.smooth) * flux

        self._frame_count += 1
        if self._frame_count <= self.warmup_frames:
            return False, flux

        onset = flux > max(1e-12, self.env * self.thr)
        return onset, flux


def detect_peaks(mags_db, freqs, top_n=5, min_distance=2):
    peak_idx = signal.find_peaks(mags_db, distance=min_distance)[0]
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
        self.start_time = None
        self.samples_read = 0

    def read_block(self):
        with self.lock:
            if self.pos >= self.f.shape[0]:
                return None

            if self.start_time is None:
                self.start_time = time.time()

            expected_elapsed = self.samples_read / self.sr
            actual_elapsed = time.time() - self.start_time
            if actual_elapsed < expected_elapsed:
                return None

            end = min(self.pos + self.blocksize, self.f.shape[0])
            block = self.f[self.pos:end, 0].copy()
            actual_size = block.size
            if block.size < self.blocksize:
                block = np.pad(block, (0, self.blocksize - block.size))

            self.pos = end
            self.samples_read += actual_size
            return block

    def close(self):
        pass


class LiveAudioSource:
    def __init__(self, sr=DEFAULT_SR, blocksize=DEFAULT_HOP, channels=1, device=None):
        self.sr = sr
        self.blocksize = blocksize
        self.channels = channels
        self.buffer = collections.deque()
        self.lock = threading.Lock()
        try:
            self.stream = sd.InputStream(samplerate=self.sr, blocksize=self.blocksize,
                                         channels=self.channels, callback=self._callback,
                                         device=device)
            self.stream.start()
        except Exception as e:
            raise RuntimeError(
                f"Could not open an audio input device (sr={sr}, channels={channels}, "
                f"device={device!r}). Check that a microphone is connected and available. "
                f"Original error: {e}"
            ) from e

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

    def close(self):
        try:
            if self.stream is not None and self.stream.active:
                self.stream.stop()
            if self.stream is not None:
                self.stream.close()
        except Exception as e:
            print("Error closing audio stream:", e, file=sys.stderr)


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
        self.setWindowTitle("DFT Visualizer — Full Edition (PATCHED)")
        self.audio_source = audio_source
        self.sr = sr
        self.fft_size = fft_size
        self.is_closing = False  # ✓ PATCHED: Race condition protection

        self.hop = max(MIN_HOP, min(hop, fft_size))

        self.spec_len = spec_len
        self.buffer = np.zeros(0, dtype=np.float32)
        self.window_name = 'hann'
        self.window_fn = WINDOWS[self.window_name]
        self.smooth_alpha = DEFAULT_SMOOTH
        self.db_floor = DEFAULT_DB_FLOOR
        self.gain = 1.0
        self.peak_count = 6
        self.onset_thr = 4.0
        self.colormap_name = 'viridis'
        self.fft_changing = False  # ✓ PATCHED: Protect FFT resize

        self.freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sr)
        self.sgram = np.full((self.freqs.size, self.spec_len), self.db_floor, dtype=float)
        self.smooth_spec = None
        self.onset_detector = OnsetDetector(thr=self.onset_thr)
        self.recorder = Recorder(self.sr)

        self._build_ui()
        self._apply_colormap()

        # Audio analysis pacing timer
        self.timer = QtCore.QTimer()
        interval = max(5, int(self.hop / float(self.sr) * 1000.0) // 2)
        self.timer.setInterval(interval)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

        # Dedicated onset visual flash reset timer
        self.onset_flash_timer = QtCore.QTimer()
        self.onset_flash_timer.setSingleShot(True)
        self.onset_flash_timer.timeout.connect(self._reset_onset_flash)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout()
        central.setLayout(layout)

        ctrl_row = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl_row)

        self.fft_combo = QtWidgets.QComboBox()
        choices = list(FFT_CHOICES)
        if self.fft_size not in choices:
            choices.append(self.fft_size)
            choices.sort()
        for n in choices:
            self.fft_combo.addItem(str(n))
        self.fft_combo.setCurrentText(str(self.fft_size))
        self.fft_combo.currentTextChanged.connect(self._on_fft_change)
        ctrl_row.addWidget(QtWidgets.QLabel("FFT size:"))
        ctrl_row.addWidget(self.fft_combo)

        self.hop_spin = QtWidgets.QSpinBox()
        self.hop_spin.setRange(MIN_HOP, self.fft_size)
        self.hop_spin.setValue(self.hop)
        self.hop_spin.valueChanged.connect(self._on_hop_change)
        ctrl_row.addWidget(QtWidgets.QLabel("Hop:"))
        ctrl_row.addWidget(self.hop_spin)

        self.window_combo = QtWidgets.QComboBox()
        for w in WINDOWS.keys():
            self.window_combo.addItem(w)
        self.window_combo.setCurrentText(self.window_name)
        self.window_combo.currentTextChanged.connect(self._on_window_change)
        ctrl_row.addWidget(QtWidgets.QLabel("Window:"))
        ctrl_row.addWidget(self.window_combo)

        self.smooth_slider = QtWidgets.QSlider(QT_HORIZONTAL)
        self.smooth_slider.setRange(0, 99)
        self.smooth_slider.setValue(int(self.smooth_alpha * 99))
        self.smooth_slider.valueChanged.connect(self._on_smooth_change)
        ctrl_row.addWidget(QtWidgets.QLabel("Smoothing:"))
        ctrl_row.addWidget(self.smooth_slider)

        self.gain_slider = QtWidgets.QSlider(QT_HORIZONTAL)
        self.gain_slider.setRange(1, 400)
        self.gain_slider.setValue(int(self.gain * 100))
        self.gain_slider.valueChanged.connect(self._on_gain_change)
        ctrl_row.addWidget(QtWidgets.QLabel("Gain:"))
        ctrl_row.addWidget(self.gain_slider)

        ctrl_row2 = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl_row2)

        self.dbfloor_slider = QtWidgets.QSlider(QT_HORIZONTAL)
        self.dbfloor_slider.setRange(-140, -20)
        self.dbfloor_slider.setValue(int(self.db_floor))
        self.dbfloor_slider.valueChanged.connect(self._on_dbfloor_change)
        ctrl_row2.addWidget(QtWidgets.QLabel("dB floor:"))
        ctrl_row2.addWidget(self.dbfloor_slider)

        # ✓ PATCHED: Added onset threshold slider
        self.onset_thr_slider = QtWidgets.QSlider(QT_HORIZONTAL)
        self.onset_thr_slider.setRange(1, 30)
        self.onset_thr_slider.setValue(int(self.onset_thr * 2))
        self.onset_thr_slider.valueChanged.connect(self._on_onset_thr_change)
        ctrl_row2.addWidget(QtWidgets.QLabel("Onset Thr:"))
        ctrl_row2.addWidget(self.onset_thr_slider)

        self.colormap_combo = QtWidgets.QComboBox()
        for c in COLORMAPS:
            self.colormap_combo.addItem(c)
        self.colormap_combo.setCurrentText(self.colormap_name)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_change)
        ctrl_row2.addWidget(QtWidgets.QLabel("Colormap:"))
        ctrl_row2.addWidget(self.colormap_combo)

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

        # ✓ PATCHED: Status bar for feedback
        self.statusBar().showMessage("Ready")

        plot_split = QtWidgets.QSplitter(QT_VERTICAL)
        layout.addWidget(plot_split)

        self.plot_widget = pg.PlotWidget(title="Spectrum")
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_widget.setLabel('left', 'Magnitude', units='dB')
        self.plot_widget.showGrid(True, True, alpha=0.3)
        self.spectrum_curve = self.plot_widget.plot([], pen='y')
        plot_split.addWidget(self.plot_widget)

        self.peak_text_items = []

        self.img_view = pg.ImageView()
        self.img_view.ui.roiBtn.hide()
        self.img_view.ui.menuBtn.hide()
        plot_split.addWidget(self.img_view)

    def _apply_colormap(self):
        try:
            cmap = pg.colormap.get(self.colormap_name)
            self.img_view.setColorMap(cmap)
            self.statusBar().showMessage(f"Colormap: {self.colormap_name}")
        except Exception as e:
            self.statusBar().showMessage(f"Colormap error: {e}")

    def _on_fft_change(self, txt):
        try:
            n = int(txt)
            self.fft_changing = True  # ✓ PATCHED: Protect resize
            self.fft_size = n
            self.hop_spin.setMaximum(n)
            if self.hop > n:
                self.hop = n
                self.hop_spin.setValue(n)
            self._recalc_freqs()
            self.fft_changing = False
        except Exception as e:
            self.statusBar().showMessage(f"FFT change error: {e}")
            self.fft_changing = False

    def _on_hop_change(self, val):
        self.hop = max(MIN_HOP, min(int(val), self.fft_size))
        interval = max(5, int(self.hop / float(self.sr) * 1000.0) // 2)
        self.timer.setInterval(interval)

    def _on_window_change(self, txt):
        self.window_name = txt
        self.window_fn = WINDOWS.get(txt, WINDOWS['hann'])

    def _on_smooth_change(self, val):
        self.smooth_alpha = val / 99.0

    def _on_gain_change(self, val):
        self.gain = val / 100.0

    def _on_dbfloor_change(self, val):
        self.db_floor = float(val)
        self.plot_widget.setYRange(self.db_floor, DEFAULT_DB_CEILING)

    # ✓ PATCHED: Added onset threshold control
    def _on_onset_thr_change(self, val):
        self.onset_thr = val / 2.0
        self.onset_detector.thr = self.onset_thr

    def _on_colormap_change(self, txt):
        self.colormap_name = txt
        self._apply_colormap()

    def _on_snapshot(self):
        """v3.1 Fix: Use the explicit pyqtgraph ImageExporter targetting 
        the internal ViewBox of the ImageView structure (`self.img_view.view`)"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"spectrogram_snapshot_{ts}.png"
        try:
            exporter = pg.exporters.ImageExporter(self.img_view.view)
            exporter.export(fname)
            self.statusBar().showMessage(f"✓ Snapshot: {fname}")
        except Exception as e:
            self.statusBar().showMessage(f"✗ Snapshot failed: {e}")

    def _on_record_toggle(self, toggled):
        if toggled:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"recording_{ts}.wav"
            self.recorder.start(fname)
            self.btn_record.setText("Stop Recording")
            self.statusBar().showMessage(f"▶ Recording: {fname}")
        else:
            self.recorder.stop()
            self.btn_record.setText("Record")
            self.statusBar().showMessage("Recording stopped")

    def _on_save_preset(self):
        preset = {
            'fft_size': self.fft_size,
            'hop': self.hop,
            'window': self.window_name,
            'smooth': self.smooth_alpha,
            'gain': self.gain,
            'spec_len': self.spec_len,
            'db_floor': self.db_floor,
            'colormap': self.colormap_name,
            'onset_thr': self.onset_thr,
        }
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Preset", "", "JSON Files (*.json)")
        if fname:
            try:
                with open(fname, 'w') as f:
                    json.dump(preset, f, indent=2)
                self.statusBar().showMessage(f"✓ Preset saved: {fname}")
            except Exception as e:
                self.statusBar().showMessage(f"✗ Save failed: {e}")

    def _on_load_preset(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON Files (*.json)")
        if fname:
            try:
                with open(fname, 'r') as f:
                    preset = json.load(f)
                self.fft_size = int(preset.get('fft_size', self.fft_size))
                self.hop_spin.setMaximum(self.fft_size)
                self.hop = max(MIN_HOP, min(int(preset.get('hop', self.hop)), self.fft_size))
                self.window_name = preset.get('window', self.window_name)
                self.window_fn = WINDOWS.get(self.window_name, WINDOWS['hann'])
                self.smooth_alpha = float(preset.get('smooth', self.smooth_alpha))
                self.gain = float(preset.get('gain', self.gain))
                self.spec_len = int(preset.get('spec_len', self.spec_len))
                self.db_floor = float(preset.get('db_floor', self.db_floor))
                self.colormap_name = preset.get('colormap', self.colormap_name)
                # ✓ PATCHED: Load onset threshold
                self.onset_thr = float(preset.get('onset_thr', self.onset_thr))

                if str(self.fft_size) not in [self.fft_combo.itemText(i) for i in range(self.fft_combo.count())]:
                    self.fft_combo.addItem(str(self.fft_size))
                self.fft_combo.setCurrentText(str(self.fft_size))
                self.hop_spin.setValue(self.hop)
                self.window_combo.setCurrentText(self.window_name)
                self.smooth_slider.setValue(int(self.smooth_alpha * 99))
                self.gain_slider.setValue(int(self.gain * 100))
                self.dbfloor_slider.setValue(int(self.db_floor))
                self.onset_thr_slider.setValue(int(self.onset_thr * 2))
                self.colormap_combo.setCurrentText(self.colormap_name)
                self._apply_colormap()
                self.statusBar().showMessage(f"✓ Preset loaded: {fname}")
                self._recalc_freqs()
            except Exception as e:
                self.statusBar().showMessage(f"✗ Load failed: {e}")

    def _recalc_freqs(self):
        self.freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sr)
        self.sgram = np.full((self.freqs.size, self.spec_len), self.db_floor, dtype=float)
        self.smooth_spec = None
        self.buffer = np.zeros(0, dtype=np.float32)
        self.onset_detector = OnsetDetector(thr=self.onset_thr)

    def _on_timer(self):
        # ✓ PATCHED: Check is_closing flag
        if self.is_closing:
            return

        block = self.audio_source.read_block()
        if block is None:
            return

        self.recorder.push(block)
        self.buffer = np.concatenate([self.buffer, block])

        if self.buffer.size < self.fft_size:
            return

        frame = self.buffer[:self.fft_size]
        self.buffer = self.buffer[self.hop:]

        mag = compute_spectrum(frame * self.gain, self.fft_size, self.window_fn)
        onset, flux = self.onset_detector.feed(mag)

        mag_db = db_amp(mag, floor_db=self.db_floor)
        if self.smooth_spec is None:
            self.smooth_spec = mag_db
        else:
            self.smooth_spec = self.smooth_alpha * self.smooth_spec + (1.0 - self.smooth_alpha) * mag_db

        self.spectrum_curve.setData(self.freqs, self.smooth_spec)
        self.plot_widget.setYRange(self.db_floor, DEFAULT_DB_CEILING)

        peaks = detect_peaks(self.smooth_spec, self.freqs, top_n=max(1, min(self.peak_count, 20)))  # ✓ PATCHED: Bounds check

        # ✓ PATCHED: Better exception handling for peak cleanup
        for it in self.peak_text_items:
            try:
                self.plot_widget.removeItem(it)
            except Exception:
                pass
        self.peak_text_items = []

        for (f, dbv, idx) in peaks:
            if f <= 0:
                continue  # ✓ PATCHED: Use continue instead of return
            note, cents, _ = freq_to_note_name(f)
            txt = f"{f:.1f} Hz\n{note} {cents:+.1f}c"
            ti = pg.TextItem(txt, anchor=(0.5, 1.0), color='w')
            ti.setPos(math.log10(f), dbv)

            self.plot_widget.addItem(ti)
            self.peak_text_items.append(ti)

        # ✓ PATCHED: Check if FFT size is being changed
        if not self.fft_changing:
            self.sgram = np.roll(self.sgram, -1, axis=1)
            self.sgram[:, -1] = mag_db
            self.img_view.setImage(self.sgram, autoLevels=False, autoRange=False)

        if onset:
            self.plot_widget.setBackground('#400000')
            self.setWindowTitle("DFT Visualizer — ONSET!")
            self.onset_flash_timer.start(120)

    def _reset_onset_flash(self):
        self.plot_widget.setBackground('k')
        self.setWindowTitle("DFT Visualizer — Full Edition (PATCHED)")

    def closeEvent(self, event):
        # ✓ PATCHED: Set flag first, then stop timer
        self.is_closing = True
        try:
            self.timer.stop()
            self.onset_flash_timer.stop()
        except Exception:
            pass
        try:
            if self.recorder.recording:
                self.recorder.stop()
        except Exception as e:
            print("Error stopping recorder on close:", e, file=sys.stderr)
        try:
            close_fn = getattr(self.audio_source, 'close', None)
            if callable(close_fn):
                close_fn()
        except Exception as e:
            print("Error closing audio source on close:", e, file=sys.stderr)
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="DFT Audio Visualizer — Full Edition (PATCHED)")
    parser.add_argument('--mode', choices=['live', 'file'], required=True)
    parser.add_argument('--file', type=str, help='If mode=file, path to audio file')
    parser.add_argument('--sr', type=int, default=DEFAULT_SR)
    parser.add_argument('--fft', type=int, default=DEFAULT_FFT)
    parser.add_argument('--hop', type=int, default=DEFAULT_HOP)
    parser.add_argument('--spec_len', type=int, default=DEFAULT_SPEC_LEN)
    args = parser.parse_args()

    if args.fft <= 0:
        print("Error: --fft must be a positive integer")
        sys.exit(1)
    if args.hop <= 0:
        print("Error: --hop must be a positive integer")
        sys.exit(1)

    clamped_hop = max(MIN_HOP, min(args.hop, args.fft))

    if args.mode == 'file':
        if not args.file:
            print("Error: --file required when mode=file")
            sys.exit(1)
        try:
            src = FileAudioSource(args.file, blocksize=clamped_hop)
        except Exception as e:
            print(f"Error: could not open audio file '{args.file}': {e}")
            sys.exit(1)
        sr = src.sr
    else:
        try:
            src = LiveAudioSource(sr=args.sr, blocksize=clamped_hop)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)
        sr = args.sr

    app = QtWidgets.QApplication(sys.argv)
    vis = VisualizerApp(src, sr, fft_size=args.fft, hop=clamped_hop, spec_len=args.spec_len)
    vis.resize(1000, 800)
    vis.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
