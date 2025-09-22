# DFT Audio Visualizer (Lean & Mean)

This project is a **minimal scaffold** for a DFT/FFT-based audio visualizer in Python.  
You just plug in the full script, install dependencies, and blaze.

## Quickstart

1. Navigate:
   ```bash
   cd dft_visualizer
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Replace `dft_visualizer.py` placeholder with the full script provided by ChatGPT.

4. Run with microphone input:
   ```bash
   python dft_visualizer.py --device live
   ```

5. Or run with an audio file:
   ```bash
   python dft_visualizer.py --file examples/my_track.wav
   ```

## Features (once full script is added)
- Real-time FFT spectrum (log frequency, dB magnitude)
- Scrolling spectrogram
- Works with mic or file input

## Requirements
- numpy
- sounddevice
- soundfile
- matplotlib

---
Lean. Mean. Blazing.
