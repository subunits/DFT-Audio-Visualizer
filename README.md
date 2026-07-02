# DFT Audio Visualizer — Full Edition

A real-time DFT/FFT-based audio visualizer with a PyQt6 + pyqtgraph GUI. Works with
either live microphone input or a pre-recorded audio file.

## Quickstart

1. Navigate:
   ```bash
   cd dft_visualizer
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run with microphone input:
   ```bash
   python dft_visualizer.py --mode live
   ```

4. Or run with an audio file:
   ```bash
   python dft_visualizer.py --mode file --file examples/my_track.wav
   ```

## CLI Options

| Flag         | Description                                  | Default |
|--------------|-----------------------------------------------|---------|
| `--mode`     | `live` or `file` (required)                    | —       |
| `--file`     | Path to audio file (required if `--mode file`) | —       |
| `--sr`       | Sample rate, used in `live` mode               | 44100   |
| `--fft`      | FFT window size                                | 4096    |
| `--hop`      | Hop size between frames                        | 1024    |
| `--spec_len` | Number of frames kept in the spectrogram       | 300     |

## Features

- **Real-time FFT spectrum** — log-frequency, dB-magnitude plot, updated on a timer
  tied to the hop size.
- **Scrolling spectrogram** — rolling image view of magnitude (dB) over time.
- **Peak detection with note names** — labels the top spectral peaks with frequency
  and nearest musical note (A4 = 440 Hz), including cents offset.
- **Onset detection** — spectral-flux based; flashes the window on detected onsets.
- **Live GUI controls** — adjust FFT size, hop, window function (Hann/Hamming/
  Blackman/rectangular), smoothing, and gain while running.
- **Recording** — toggle a Record button to capture live/processed audio to a WAV file.
- **Snapshot export** — save the current spectrogram view as a PNG.
- **Presets** — save and load your control settings (FFT size, hop, window,
  smoothing, gain, spectrogram length) as JSON.

## Requirements

- numpy
- scipy
- sounddevice
- soundfile
- pyqtgraph
- PyQt6 (falls back to PyQt5 if PyQt6 isn't available)

## Notes

- Requires a GUI environment (PyQt6/PyQt5) — this is not a headless/CLI-only tool.
- File-mode input is downmixed to mono automatically if the source is stereo.
