# Copyright (C) 2025 CRCODE22
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
import numpy as np
from scipy.fft import fft
from scipy.signal import butter, lfilter
import soundfile as sf
import os
from datetime import datetime
import gradio as gr
import matplotlib.pyplot as plt
import re

# Default parameters for FSK modulation/demodulation
SAMPLE_RATE = 44100  # Audio sample rate (Hz)
BIT_DURATION = 0.1   # Duration of each bit (seconds), 10 bits/second
PREAMBLE_BITS = "10101010"  # 8-bit preamble for signal alignment

# Step 1: Generate FSK audio from text with a preamble
def generate_fsk_audio(text, freq_0, freq_1, output_file="test_fsk_audio.wav"):
    if not text:
        return None, "Error: Please enter a text message to encode."
    try:
        freq_0 = float(freq_0)
        freq_1 = float(freq_1)
        if freq_0 <= 0 or freq_1 <= 0 or freq_0 >= SAMPLE_RATE/2 or freq_1 >= SAMPLE_RATE/2:
            return None, f"Error: Frequencies must be between 0 and {SAMPLE_RATE/2} Hz (Nyquist limit)."
        if freq_0 == freq_1:
            return None, "Error: Frequencies for '0' and '1' must be different."
    except ValueError:
        return None, "Error: Frequencies must be valid numbers."

    if os.path.exists(output_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(output_file)
        new_file_name = f"{base}_{timestamp}{ext}"
        os.rename(output_file, new_file_name)
        print(f"Renamed existing file to: {new_file_name}")

    # Convert text to binary
    binary_str = PREAMBLE_BITS  # Add preamble
    for char in text:
        binary_char = format(ord(char), '08b')  # 8-bit ASCII
        binary_str += binary_char
    
    # Generate FSK signal
    samples_per_bit = int(SAMPLE_RATE * BIT_DURATION)
    audio_signal = []
    t = np.linspace(0, BIT_DURATION, samples_per_bit, endpoint=False)
    
    for bit in binary_str:
        freq = freq_0 if bit == '0' else freq_1
        signal = 0.5 * np.sin(2 * np.pi * freq * t)
        audio_signal.extend(signal)
    
    # Add 1 second of leading silence
    silence_samples = int(SAMPLE_RATE * 1.0)
    audio_signal = [0] * silence_samples + audio_signal
    
    audio_signal = np.array(audio_signal)
    sf.write(output_file, audio_signal, SAMPLE_RATE)

    # Log the encoding operation
    log_entry = (f"Encoded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                 f"Text: {text}\n"
                 f"Frequency for '0': {freq_0} Hz\n"
                 f"Frequency for '1': {freq_1} Hz\n"
                 f"Output File: {output_file}\n"
                 f"Binary (with preamble): {binary_str}\n\n")
    with open("encode_log.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

    return output_file, f"Generated FSK audio file: {output_file}"

# Step 2: Generate a test tone for frequency detection
def generate_test_tone(test_freq, duration=10.0, output_file="test_tone.wav"):
    try:
        test_freq = float(test_freq)
        if test_freq <= 0 or test_freq >= SAMPLE_RATE/2:
            return None, f"Error: Test frequency must be between 0 and {SAMPLE_RATE/2} Hz (Nyquist limit)."
    except ValueError:
        return None, "Error: Test frequency must be a valid number."

    if os.path.exists(output_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(output_file)
        new_file_name = f"{base}_{timestamp}{ext}"
        os.rename(output_file, new_file_name)
        print(f"Renamed existing file to: {new_file_name}")

    num_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    audio_signal = 0.5 * np.sin(2 * np.pi * test_freq * t)
    
    sf.write(output_file, audio_signal, SAMPLE_RATE)
    return output_file, f"Generated test tone: {output_file} (Note: Frequencies below 20 Hz may be inaudible but should be visible in a waveform viewer like GoldWave)."

# Step 3: Detect the dominant frequency in a recorded audio file
def detect_dominant_frequency(audio_data, freq_range_min=400, freq_range_max=600):
    chunk_size = int(SAMPLE_RATE * 1.0)  # 1 second
    chunk = audio_data[:chunk_size]
    fft_result = fft(chunk)
    freqs = np.fft.fftfreq(len(chunk), 1 / SAMPLE_RATE)
    fft_magnitude = np.abs(fft_result)
    
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
    freq_range = (positive_freqs >= freq_range_min) & (positive_freqs <= freq_range_max)
    if not np.any(freq_range):
        return None, "No frequencies found in the specified range."
    
    # Debug: Log the top 5 frequencies and their magnitudes
    freq_range_indices = np.where(freq_range)[0]
    top_indices = np.argsort(positive_magnitude[freq_range_indices])[::-1][:5]
    debug_info = "Top 5 frequencies in range:\n"
    for idx in top_indices:
        freq = positive_freqs[freq_range_indices][idx]
        mag = positive_magnitude[freq_range_indices][idx]
        debug_info += f"Freq: {freq:.2f} Hz, Magnitude: {mag:.2f}\n"
    
    dominant_freq = positive_freqs[freq_range][np.argmax(positive_magnitude[freq_range])]
    return dominant_freq, debug_info

# Step 4: Calculate frequency difference for encryption
def calculate_frequency_difference(audio_file, specified_freq, freq_range_min=400, freq_range_max=600):
    try:
        specified_freq = float(specified_freq)
        if specified_freq <= 0 or specified_freq >= SAMPLE_RATE/2:
            return f"Error: Specified frequency must be between 0 and {SAMPLE_RATE/2} Hz (Nyquist limit).", None
    except ValueError:
        return "Error: Specified frequency must be a valid number.", None

    audio_data, sample_rate = sf.read(audio_file)
    if sample_rate != SAMPLE_RATE:
        return f"Error: Sample rate must be {SAMPLE_RATE} Hz, got {sample_rate} Hz.", None
    
    # Use read_audio to select the stronger channel
    audio_data, left_max, right_max = read_audio(audio_file, channel_strategy="stronger")

    result = detect_dominant_frequency(audio_data, freq_range_min, freq_range_max)
    if isinstance(result, tuple):
        dominant_freq, debug_info = result
    else:
        return result, None  # Error message

    if dominant_freq is None:
        return "Error: No dominant frequency detected.", None

    freq_shift = dominant_freq - specified_freq
    suggested_freq_0 = 1000 + freq_shift
    suggested_freq_1 = 2000 + freq_shift

    spectrogram_img = generate_spectrogram(audio_file, freq_min=freq_range_min, freq_max=freq_range_max)

    return (f"Specified Frequency: {specified_freq:.2f} Hz\n"
            f"Detected Frequency: {dominant_freq:.2f} Hz\n"
            f"Frequency Shift: {freq_shift:.2f} Hz\n"
            f"Suggested FSK Frequencies for Encryption:\n"
            f"FREQ_0 = {suggested_freq_0:.2f} Hz\n"
            f"FREQ_1 = {suggested_freq_1:.2f} Hz\n\n"
            f"Debug Info:\n{debug_info}"), spectrogram_img

# Step 5: Read the recorded audio file for decoding
def read_audio(file_path, channel_strategy="stronger"):
    audio_data, sample_rate = sf.read(file_path)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Sample rate must be {SAMPLE_RATE} Hz, got {sample_rate} Hz")
    
    # Handle stereo or mono audio
    if len(audio_data.shape) > 1:  # Stereo
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]
        left_max = np.max(np.abs(left_channel))
        right_max = np.max(np.abs(right_channel))
        
        if channel_strategy == "stronger":
            if left_max >= right_max:
                selected_channel = left_channel
            else:
                selected_channel = right_channel
        elif channel_strategy == "average":
            selected_channel = (left_channel + right_channel) / 2
        else:
            raise ValueError(f"Unknown channel strategy: {channel_strategy}")
    else:  # Mono
        selected_channel = audio_data
        left_max = np.max(np.abs(audio_data))
        right_max = None
    
    # Normalize the selected channel to 0.5
    max_amplitude = np.max(np.abs(selected_channel))
    if max_amplitude > 0:
        selected_channel = selected_channel * (0.5 / max_amplitude)
    
    return selected_channel, left_max, right_max

# Step 6: Apply a high-pass filter to remove low-frequency noise
def high_pass_filter(audio_data, cutoff_freq=50.0, fs=SAMPLE_RATE):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(5, normal_cutoff, btype='high', analog=False)
    filtered_data = lfilter(b, a, audio_data)
    return filtered_data

# Step 7: Automatically detect FSK frequencies
def auto_detect_fsk_frequencies(audio_data, freq_range_min=500, freq_range_max=2500):
    samples_per_bit = int(SAMPLE_RATE * BIT_DURATION)
    # Find signal start first to align windows
    start_idx, _ = find_signal_start(audio_data, samples_per_bit)
    audio_data = audio_data[start_idx:]
    
    num_bits = len(audio_data) // samples_per_bit
    freqs_detected = []

    window = np.hanning(samples_per_bit)
    for i in range(min(num_bits, 50)):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        chunk = audio_data[start:end]
        if len(chunk) != samples_per_bit:
            continue
        chunk = chunk * window
        fft_result = fft(chunk)
        freqs = np.fft.fftfreq(len(chunk), 1 / SAMPLE_RATE)
        fft_magnitude = np.abs(fft_result)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
        freq_range = (positive_freqs >= freq_range_min) & (positive_freqs <= freq_range_max)
        if not np.any(freq_range):
            continue
        dominant_freq = positive_freqs[freq_range][np.argmax(positive_magnitude[freq_range])]
        freqs_detected.append(dominant_freq)

    if len(freqs_detected) < 2:
        return None, None, "Error: Could not detect FSK frequencies."

    freqs_detected = np.array(freqs_detected)
    threshold = np.median(freqs_detected)
    freq_0 = np.mean(freqs_detected[freqs_detected < threshold])
    freq_1 = np.mean(freqs_detected[freqs_detected >= threshold])
    return freq_0, freq_1, f"Auto-detected FSK Frequencies: FREQ_0 = {freq_0:.2f} Hz, FREQ_1 = {freq_1:.2f} Hz"

# Step 8: Find the start of the FSK signal using the preamble
def find_signal_start(audio_data, samples_per_bit, energy_threshold=0.00001, min_consecutive_windows=2):
    debug_info = []
    consecutive_above_threshold = 0
    start_idx = 0
    step_size = samples_per_bit // 4  # Step by 1/4 of a bit window
    initial_windows = 5  # Check first 5 windows with a lower threshold
    
    # Dynamically adjust the energy threshold based on the maximum amplitude
    max_amplitude = np.max(np.abs(audio_data))
    if max_amplitude > 0:
        expected_energy = 0.125  # For amplitude 0.5
        actual_energy = expected_energy * (max_amplitude / 0.5) ** 2
        energy_threshold = energy_threshold * (actual_energy / expected_energy)
        initial_threshold = energy_threshold * 0.5
    else:
        initial_threshold = energy_threshold * 0.5

    debug_info.append(f"Adjusted energy_threshold: {energy_threshold:.6f}")
    debug_info.append(f"Adjusted initial_threshold: {initial_threshold:.6f}")

    for i in range(0, len(audio_data) - samples_per_bit, step_size):
        chunk = audio_data[i:i + samples_per_bit]
        if len(chunk) != samples_per_bit:
            continue
        energy = np.mean(chunk ** 2)
        debug_info.append(f"Window {i//step_size}: Energy = {energy:.6f}")
        
        current_threshold = initial_threshold if (i // step_size) < initial_windows else energy_threshold
        
        if energy > current_threshold:
            consecutive_above_threshold += 1
            if consecutive_above_threshold == 1:
                start_idx = i
            if consecutive_above_threshold >= min_consecutive_windows:
                fine_tune_start = max(0, start_idx - samples_per_bit)
                fine_tune_end = start_idx + samples_per_bit
                max_energy = 0
                best_start = start_idx
                for j in range(fine_tune_start, fine_tune_end, step_size // 2):
                    chunk = audio_data[j:j + samples_per_bit]
                    if len(chunk) != samples_per_bit:
                        continue
                    energy = np.mean(chunk ** 2)
                    if energy > max_energy:
                        max_energy = energy
                        best_start = j
                best_start = (best_start // samples_per_bit) * samples_per_bit
                debug_info.append(f"Signal start detected at sample {best_start} (energy = {max_energy:.6f})")
                return best_start, "\n".join(debug_info)
        else:
            consecutive_above_threshold = 0
    debug_info.append("No signal start detected; starting at sample 0")
    return 0, "\n".join(debug_info)

# Step 9: Demodulate FSK signal to binary
def demodulate_fsk(audio_data, freq_0=None, freq_1=None, expected_bits=None):
    # Apply high-pass filter to remove low-frequency noise
    audio_data = high_pass_filter(audio_data, cutoff_freq=50.0)

    # Automatically detect FSK frequencies if not provided
    auto_freq_status = ""
    if freq_0 is None or freq_1 is None:
        detected_freq_0, detected_freq_1, status = auto_detect_fsk_frequencies(audio_data)
        if detected_freq_0 is None:
            return [], status
        freq_0, freq_1 = detected_freq_0, detected_freq_1
        auto_freq_status = status

    freq_threshold = (freq_0 + freq_1) / 2
    samples_per_bit = int(SAMPLE_RATE * BIT_DURATION)
    
    # Find the start of the signal
    start_idx, debug_info = find_signal_start(audio_data, samples_per_bit)
    
    # Adjust audio_data to start at the signal
    audio_data = audio_data[start_idx:]
    
    # Calculate the number of complete bit windows
    num_bits = len(audio_data) // samples_per_bit
    if expected_bits:
        num_bits = min(num_bits, expected_bits)
    binary_data = []

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(samples_per_bit)

    for i in range(num_bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        chunk = audio_data[start:end]
        if len(chunk) != samples_per_bit:
            break
        chunk = chunk * window

        fft_result = fft(chunk)
        freqs = np.fft.fftfreq(len(chunk), 1 / SAMPLE_RATE)
        fft_magnitude = np.abs(fft_result)
        
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
        freq_range = (positive_freqs >= min(freq_0, freq_1) - 50) & (positive_freqs <= max(freq_0, freq_1) + 50)
        if not np.any(freq_range):
            binary_data.append(0)
            debug_info += f"\nBit {i}: No frequency in range, defaulting to 0"
            continue
        dominant_freq = positive_freqs[freq_range][np.argmax(positive_magnitude[freq_range])]
        bit = 0 if dominant_freq < freq_threshold else 1
        binary_data.append(bit)
        debug_info += f"\nBit {i}: Dominant Freq = {dominant_freq:.2f} Hz, Threshold = {freq_threshold:.2f} Hz, Bit = {bit}"

    # Skip the preamble (8 bits)
    if len(binary_data) >= 8:
        preamble_detected = ''.join(map(str, binary_data[:8]))
        debug_info += f"\nDetected Preamble: {preamble_detected}"
        if preamble_detected == PREAMBLE_BITS:
            binary_data = binary_data[8:]
            debug_info += "\nPreamble matched, skipped 8 bits."
        else:
            debug_info += "\nPreamble mismatch, proceeding without skipping."

    debug_info = debug_info + "\n" + auto_freq_status
    return binary_data, debug_info

# Step 10: Convert binary data to text
def binary_to_text(binary_data):
    binary_str = ''.join(map(str, binary_data))
    if len(binary_str) % 8 != 0:
        binary_str = binary_str[:-(len(binary_str) % 8)]

    text = ''
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        if len(byte) == 8:
            char_code = int(byte, 2)
            text += chr(char_code)
    return binary_str, text

# Step 11: Generate waveform, spectrogram, and frequency spectrum visualizations
def generate_audio_visualizations(audio_file, output_waveform="waveform.png", output_spectrogram="spectrogram.png", output_spectrum="spectrum.png", freq_0=None, freq_1=None):
    audio_data, sample_rate = sf.read(audio_file)
    
    if len(audio_data.shape) > 1:  # Stereo
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]
    else:  # Mono
        left_channel = audio_data
        right_channel = None
    
    # Waveform
    plt.figure(figsize=(10, 6))
    t = np.linspace(0, len(left_channel) / sample_rate, len(left_channel))
    plt.subplot(2, 1, 1)
    plt.plot(t, left_channel, color='blue', label='Left Channel')
    plt.title("Waveform - Left Channel")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    
    if right_channel is not None:
        plt.subplot(2, 1, 2)
        plt.plot(t, right_channel, color='green', label='Right Channel')
        plt.title("Waveform - Right Channel")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_waveform)
    plt.close()

    # Spectrogram (zoomed to FSK frequency range)
    freq_min = min(freq_0, freq_1) - 100 if freq_0 and freq_1 else 400
    freq_max = max(freq_0, freq_1) + 100 if freq_0 and freq_1 else 1600
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.specgram(left_channel, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title("Spectrogram - Left Channel")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(freq_min, freq_max)
    plt.colorbar(label="Intensity (dB)")
    
    if right_channel is not None:
        plt.subplot(2, 1, 2)
        plt.specgram(right_channel, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
        plt.title("Spectrogram - Right Channel")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(freq_min, freq_max)
        plt.colorbar(label="Intensity (dB)")
    
    plt.tight_layout()
    plt.savefig(output_spectrogram)
    plt.close()

    # Frequency spectrum for the first few bits
    if freq_0 and freq_1:
        samples_per_bit = int(SAMPLE_RATE * BIT_DURATION)
        start_idx, _ = find_signal_start(audio_data[:, 1] if len(audio_data.shape) > 1 else audio_data, samples_per_bit)
        audio_data_selected = (audio_data[:, 1] if len(audio_data.shape) > 1 and np.max(np.abs(audio_data[:, 1])) > np.max(np.abs(audio_data[:, 0])) else audio_data[:, 0] if len(audio_data.shape) > 1 else audio_data)
        audio_data_selected = audio_data_selected[start_idx:]
        
        plt.figure(figsize=(10, 8))
        window = np.hanning(samples_per_bit)
        for i in range(min(4, len(audio_data_selected) // samples_per_bit)):
            start = i * samples_per_bit
            end = (i + 1) * samples_per_bit
            chunk = audio_data_selected[start:end]
            if len(chunk) != samples_per_bit:
                break
            chunk = chunk * window
            fft_result = fft(chunk)
            freqs = np.fft.fftfreq(len(chunk), 1 / SAMPLE_RATE)
            fft_magnitude = np.abs(fft_result)
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
            plt.subplot(4, 1, i+1)
            plt.plot(positive_freqs, positive_magnitude, label=f"Bit {i}")
            plt.title(f"Frequency Spectrum - Bit {i}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.xlim(freq_min, freq_max)
            plt.grid()
            plt.legend()
        plt.tight_layout()
        plt.savefig(output_spectrum)
        plt.close()

        return output_waveform, output_spectrogram, output_spectrum
    return output_waveform, output_spectrogram, None

# Step 12: Debug audio file for leading silence and amplitude
def debug_audio_file(audio_file):
    audio_data, sample_rate = sf.read(audio_file)  # Read without normalization for debug
    debug_info = []
    
    if len(audio_data.shape) > 1:  # Stereo
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]
        left_max = np.max(np.abs(left_channel))
        right_max = np.max(np.abs(right_channel))
        debug_info.append(f"Max Amplitude (Left Channel): {left_max:.6f}")
        debug_info.append(f"Max Amplitude (Right Channel): {right_max:.6f}")
        if left_max >= right_max:
            debug_info.append("Selected Channel: Left")
        else:
            debug_info.append("Selected Channel: Right")
    else:  # Mono
        left_channel = audio_data
        left_max = np.max(np.abs(audio_data))
        debug_info.append(f"Max Amplitude (Mono): {left_max:.6f}")
        debug_info.append("Selected Channel: Mono")
    
    samples_per_bit = int(SAMPLE_RATE * BIT_DURATION)
    threshold = 0.0001
    for i in range(0, len(left_channel) - samples_per_bit, samples_per_bit):
        chunk = left_channel[i:i + samples_per_bit]
        if len(chunk) != samples_per_bit:
            break
        amplitude = np.max(np.abs(chunk))
        if amplitude > threshold:
            debug_info.append(f"Signal detected at sample {i} (amplitude = {amplitude:.6f})")
            break
    else:
        debug_info.append("No signal detected above threshold.")
    
    return "\n".join(debug_info)

# Step 12.1: Generate spectrogram for frequency analysis
def generate_spectrogram(audio_file, output_spectrogram="test_tone_spectrogram.png", freq_min=400, freq_max=1600):
    audio_data, sample_rate = sf.read(audio_file)
    
    if len(audio_data.shape) > 1:  # Stereo
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]
    else:  # Mono
        left_channel = audio_data
        right_channel = None
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.specgram(left_channel, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title("Spectrogram of Test Tone - Left Channel")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(freq_min, freq_max)
    plt.colorbar(label="Intensity (dB)")
    
    if right_channel is not None:
        plt.subplot(2, 1, 2)
        plt.specgram(right_channel, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
        plt.title("Spectrogram of Test Tone - Right Channel")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(freq_min, freq_max)
        plt.colorbar(label="Intensity (dB)")
    
    plt.tight_layout()
    plt.savefig(output_spectrogram)
    plt.close()
    return output_spectrogram

# Step 13: Decode the FSK audio
def decode_fsk_audio(audio_file, freq_0=None, freq_1=None, expected_length=None, visualize=False):
    if audio_file is None:
        return "", "", "Error: Please upload a recorded audio file.", "", None, None, None
    
    try:
        if freq_0 is not None:
            freq_0 = float(freq_0)
            freq_1 = float(freq_1)
            if freq_0 <= 0 or freq_1 <= 0 or freq_0 >= SAMPLE_RATE/2 or freq_1 >= SAMPLE_RATE/2:
                return "", "", f"Error: Frequencies must be between 0 and {SAMPLE_RATE/2} Hz (Nyquist limit).", "", None, None, None
            if freq_0 == freq_1:
                return "", "", "Error: Frequencies for '0' and '1' must be different.", "", None, None, None
    except ValueError:
        return "", "", "Error: Frequencies must be valid numbers.", "", None, None, None

    try:
        # Debug the audio file
        audio_debug_info = debug_audio_file(audio_file)
        print(audio_debug_info)

        # Read audio and get channel amplitudes
        audio_data, left_max, right_max = read_audio(audio_file, channel_strategy="stronger")
        expected_bits = None
        if expected_length:
            expected_bits = expected_length * 8
        binary_data, debug_info = demodulate_fsk(audio_data, freq_0, freq_1, expected_bits)
        binary_str, decoded_text = binary_to_text(binary_data)

        waveform_img, spectrogram_img, spectrum_img = None, None, None
        if visualize:
            waveform_img, spectrogram_img, spectrum_img = generate_audio_visualizations(audio_file, freq_0=freq_0, freq_1=freq_1)

        debug_info = audio_debug_info + "\n\n" + debug_info
        return binary_str, decoded_text, "Decoding successful.", debug_info, waveform_img, spectrogram_img, spectrum_img
    except Exception as e:
        return "", "", f"Error: {str(e)}", "", None, None, None

# Step 14: Batch decode multiple audio files
def batch_decode_fsk_audio(audio_files, freq_0=None, freq_1=None, expected_length=None, visualize=False):
    if not audio_files:
        return [], "Error: Please upload at least one audio file."

    results = []
    for audio_file in audio_files:
        binary_str, decoded_text, status, debug_info, waveform_img, spectrogram_img, spectrum_img = decode_fsk_audio(
            audio_file, freq_0, freq_1, expected_length, visualize
        )
        # Save results for each file
        output_file = os.path.splitext(os.path.basename(audio_file))[0] + "_decoded.txt"
        debug_file = os.path.splitext(os.path.basename(audio_file))[0] + "_debug.txt"
        save_status = save_decoded_output(binary_str, decoded_text, debug_info, freq_0, freq_1, expected_length, output_file, debug_file)
        results.append({
            "File": audio_file,
            "Binary": binary_str,
            "Decoded Text": decoded_text,
            "Status": status,
            "Debug Info": debug_info,
            "Waveform": waveform_img,
            "Spectrogram": spectrogram_img,
            "Spectrum": spectrum_img,
            "Save Status": save_status
        })
    return results, "Batch decoding completed."

# Step 15: Save decoded output and debug info to files
def parse_auto_detected_frequencies(debug_info):
    pattern = r"Auto-detected FSK Frequencies: FREQ_0 = (\d+\.\d+) Hz, FREQ_1 = (\d+\.\d+) Hz"
    match = re.search(pattern, debug_info)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def save_decoded_output(binary_str, decoded_text, debug_info, freq_0, freq_1, expected_length, output_file="decoded_output.txt", debug_file="debug_log.txt"):
    if not binary_str and not decoded_text:
        return "Error: No decoded output to save."
    
    if not output_file.endswith('.txt'):
        output_file = output_file + '.txt'
    if not debug_file.endswith('.txt'):
        debug_file = debug_file + '.txt'
    if not output_file or not debug_file:
        return "Error: Output and debug file names cannot be empty."
    
    try:
        auto_freq_0, auto_freq_1 = parse_auto_detected_frequencies(debug_info) if (freq_0 is None or freq_1 is None) else (None, None)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Decoded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Decoded Binary:\n")
            f.write(binary_str + "\n\n")
            f.write("Decoded Text:\n")
            f.write(decoded_text + "\n\n")
            f.write("Decode Parameters:\n")
            f.write(f"Frequency for '0' (Hz): {freq_0 if freq_0 is not None else f'{auto_freq_0:.2f} (Auto-detected)' if auto_freq_0 else 'Auto-detected'}\n")
            f.write(f"Frequency for '1' (Hz): {freq_1 if freq_1 is not None else f'{auto_freq_1:.2f} (Auto-detected)' if auto_freq_1 else 'Auto-detected'}\n")
            f.write(f"Expected Message Length (characters): {expected_length if expected_length is not None else 'Not specified'}\n\n")
            f.write("Debug Info:\n")
            f.write(debug_info + "\n")
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write("Debug Info:\n")
            f.write(debug_info + "\n")
        
        return f"Saved decoded output to {output_file} and debug log to {debug_file}"
    except PermissionError:
        return f"Error: Permission denied when writing to {output_file} or {debug_file}."
    except OSError as e:
        return f"Error: Failed to write files ({str(e)})."
    except Exception as e:
        return f"Error: An unexpected error occurred while saving files: {str(e)}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Atlantean Crystal FSK Encoder/Decoder")
    gr.Markdown("Encode text into an FSK audio signal, decode recorded audio, and analyze frequency shifts for crystal-based encryption!")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Encode Text to FSK Audio")
            text_input = gr.Textbox(label="Text to Encode", value="HI", placeholder="Enter your message here")
            freq_0_input = gr.Number(label="Frequency for '0' (Hz)", value=1000, step=0.01)
            freq_1_input = gr.Number(label="Frequency for '1' (Hz)", value=2000, step=0.01)
            generate_button = gr.Button("Generate FSK Audio")
            audio_output = gr.Audio(label="Generated FSK Audio File")
            encode_status = gr.Textbox(label="Status")

        with gr.Column():
            gr.Markdown("## Decode Recorded FSK Audio")
            audio_input = gr.File(label="Upload Recorded Audio (WAV)", file_types=[".wav"])
            decode_freq_0_input = gr.Number(label="Frequency for '0' (Hz, leave blank for auto-detect)", value=None, step=0.01)
            decode_freq_1_input = gr.Number(label="Frequency for '1' (Hz, leave blank for auto-detect)", value=None, step=0.01)
            expected_length_input = gr.Number(label="Expected Message Length (characters)", value=5, step=1)
            visualize_checkbox = gr.Checkbox(label="Generate Waveform, Spectrogram, and Spectrum Visualizations", value=False)
            decode_button = gr.Button("Decode FSK Audio")
            decode_binary_output = gr.Textbox(label="Decoded Binary")
            decode_text_output = gr.Textbox(label="Decoded Text")
            decode_status = gr.Textbox(label="Status")
            debug_output = gr.Textbox(label="Debug Info")
            waveform_output = gr.Image(label="Waveform Visualization")
            spectrogram_output = gr.Image(label="Spectrogram Visualization")
            spectrum_output = gr.Image(label="Frequency Spectrum Visualization")
            save_button = gr.Button("Save Decoded Output and Debug Log to Files")
            output_file_input = gr.Textbox(label="Output File Name", value="decoded_output.txt")
            debug_file_input = gr.Textbox(label="Debug File Name", value="debug_log.txt")
            save_status = gr.Textbox(label="Save Status")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Batch Decode Multiple Audio Files")
            batch_audio_input = gr.File(label="Upload Multiple Recorded Audio Files (WAV)", file_types=[".wav"], file_count="multiple")
            batch_freq_0_input = gr.Number(label="Frequency for '0' (Hz, leave blank for auto-detect)", value=None, step=0.01)
            batch_freq_1_input = gr.Number(label="Frequency for '1' (Hz, leave blank for auto-detect)", value=None, step=0.01)
            batch_expected_length_input = gr.Number(label="Expected Message Length (characters)", value=5, step=1)
            batch_visualize_checkbox = gr.Checkbox(label="Generate Visualizations for Each File", value=False)
            batch_decode_button = gr.Button("Batch Decode FSK Audio")
            batch_results_output = gr.JSON(label="Batch Decode Results")
            batch_status = gr.Textbox(label="Batch Status")

        with gr.Column():
            gr.Markdown("## Frequency Difference Analysis")
            test_audio_input = gr.File(label="Upload Recorded Test Tone (WAV)", file_types=[".wav"])
            specified_freq_input = gr.Number(label="Specified Frequency (Hz)", value=7.83, step=0.01)
            freq_range_min_input = gr.Number(label="Frequency Range Min (Hz)", value=400)
            freq_range_max_input = gr.Number(label="Frequency Range Max (Hz)", value=600)
            analyze_button = gr.Button("Calculate Frequency Shift")
            freq_analysis_output = gr.Textbox(label="Frequency Analysis")
            test_spectrogram_output = gr.Image(label="Spectrogram of Test Tone")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Generate Test Tone for Frequency Detection")
            test_freq_input = gr.Number(label="Test Frequency (Hz)", value=7.83, step=0.01)
            generate_test_button = gr.Button("Generate Test Tone")
            test_tone_output = gr.Audio(label="Generated Test Tone")
            test_tone_status = gr.Textbox(label="Status")

    # Connect buttons to functions
    generate_button.click(
        fn=generate_fsk_audio,
        inputs=[text_input, freq_0_input, freq_1_input],
        outputs=[audio_output, encode_status]
    )
    decode_outputs = decode_button.click(
        fn=decode_fsk_audio,
        inputs=[audio_input, decode_freq_0_input, decode_freq_1_input, expected_length_input, visualize_checkbox],
        outputs=[decode_binary_output, decode_text_output, decode_status, debug_output, waveform_output, spectrogram_output, spectrum_output]
    )
    save_button.click(
        fn=save_decoded_output,
        inputs=[decode_binary_output, decode_text_output, debug_output, decode_freq_0_input, decode_freq_1_input, expected_length_input, output_file_input, debug_file_input],
        outputs=save_status
    )
    batch_decode_button.click(
        fn=batch_decode_fsk_audio,
        inputs=[batch_audio_input, batch_freq_0_input, batch_freq_1_input, batch_expected_length_input, batch_visualize_checkbox],
        outputs=[batch_results_output, batch_status]
    )
    analyze_button.click(
        fn=calculate_frequency_difference,
        inputs=[test_audio_input, specified_freq_input, freq_range_min_input, freq_range_max_input],
        outputs=[freq_analysis_output, test_spectrogram_output]
    )
    generate_test_button.click(
        fn=generate_test_tone,
        inputs=[test_freq_input],
        outputs=[test_tone_output, test_tone_status]
    )

# Launch the Gradio app
demo.launch()
