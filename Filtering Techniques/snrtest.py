import numpy as np
from scipy.io import wavfile

def calculate_snr(wav_file_path, noise_start_time, noise_end_time, signal_start_time, signal_end_time):
    # Read the audio file
    sample_rate, audio_data = wavfile.read(wav_file_path)

    # Convert to float for accurate calculations
    audio_data = audio_data.astype(np.float32)

    # If stereo, convert to mono by averaging the channels
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Calculate sample indices for the noise and signal segments
    noise_start_index = int(noise_start_time * sample_rate)
    noise_end_index = int(noise_end_time * sample_rate)
    signal_start_index = int(signal_start_time * sample_rate)
    signal_end_index = int(signal_end_time * sample_rate)

    # Extract noise and signal segments
    noise_segment = audio_data[noise_start_index:noise_end_index]
    signal_segment = audio_data[signal_start_index:signal_end_index]

    # Calculate power for noise and signal segments
    noise_power = np.mean(noise_segment ** 2)
    signal_power = np.mean(signal_segment ** 2)

    # Ensure noise power is not zero to avoid division by zero errors
    if noise_power == 0:
        raise ValueError("Noise power is zero; cannot compute SNR.")

    # Calculate SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

# Example usage
wav_file_path = '/Users/shaanmehta/Downloads/band_pass_filtered.wav'  # Replace with your actual file path
snr_value = calculate_snr(wav_file_path, noise_start_time=0.1, noise_end_time=0.25, signal_start_time=0.4, signal_end_time=1.25)
print(f"Signal-to-Noise Ratio (SNR) of the audio file is: {snr_value:.2f} dB")
