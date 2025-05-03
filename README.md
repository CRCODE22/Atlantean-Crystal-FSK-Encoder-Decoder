# CrystalFSK

A Python-based tool to encode text into Frequency Shift Keying (FSK) audio signals and decode recorded audio, inspired by Atlantean crystal communication. Built with Gradio for an interactive web interface.

## Features
- Encode text to FSK audio with customizable frequencies.
- Decode FSK audio with automated preprocessing (900–1600 Hz bandpass filter, 75% amplification, signal trimming).
- Generate visualizations: waveform, spectrogram, frequency spectrum.
- Supports batch decoding of multiple audio files.
- Optimized for CTF challenges and crystal-based experiments.
- Licensed under GNU General Public License v3.0.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CRCODE22/CrystalFSK.git
   cd CrystalFSK
