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
   
## Install dependencies
pip install numpy scipy soundfile gradio matplotlib

## Run the app
python fsk_gradio.py

## Usage

Encode: Enter text for example: "I am a smokey quartz crystal!" , set frequencies (e.g., 1000 Hz for '0', 1500 Hz for '1'), and generate an FSK audio file.

Decode: Upload a 44100 Hz WAV file, use defaults (1000 Hz, 1500 Hz, 86 characters), and decode the message.

Visualizations: Enable waveform, spectrogram, and spectrum for signal analysis.

Batch Decode: Process multiple audio files at once.

## Example

Decode the recorded audio to retrieve:
I am a smokey quartz crystal! If you used the text from the example text above ^
