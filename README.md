# DFT Audio Visualizer

This project provides a DFT/FFT-based audio visualizer in Python.  
It supports live microphone input and audio file playback with real-time spectrum and spectrogram visualization.

## Features
- Real-time FFT spectrum
- Scrolling spectrogram
- Log frequency scaling
- dB magnitude display

## Usage
```bash
pip install -r requirements.txt
python dft_visualizer.py --device live
python dft_visualizer.py --file examples/my_track.wav
```

## Requirements
- numpy
- sounddevice
- soundfile
- matplotlib
