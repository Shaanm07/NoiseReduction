# Note, the following code will only work for MY file path. 
# To implement the same task, you must create your own filepath.

import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.fftpack as fft
from scipy.signal import medfilt

# Load the audio file
y, sr = librosa.load("/Users/shaanmehta/Downloads/noisy.wav", sr=None)

# Perform Short-Time Fourier Transform (STFT)
S_full, phase = librosa.magphase(librosa.stft(y))

# Estimate noise power
noise_power = np.mean(S_full[:, :int(sr*0.1)], axis=1)

# Create a mask to filter out noise
mask = S_full > noise_power[:, None]
mask = mask.astype(float)

# Apply median filter to the mask
mask = medfilt(mask, kernel_size=(1, 5))

# Apply the mask to the magnitude spectrum
S_clean = S_full * mask

# Reconstruct the time-domain signal from the clean magnitude and original phase
y_clean = librosa.istft(S_clean * phase)

# Write the cleaned audio to a file
sf.write("/Users/shaanmehta/Downloads/clean.wav", y_clean, sr)

# Plotting the original, isolated noise, and clean audio

# Function to plot audio signal
def plot_audio(signal, sr, title, subplot):
    plt.subplot(subplot)
    plt.plot(np.linspace(0, len(signal) / sr, num=len(signal)), signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

# Plot original noisy audio
plt.figure(figsize=(14, 10))

plot_audio(y, sr, "Original Noisy Audio", 311)

# Calculate the isolated noise
S_noisy = S_full * (1 - mask)
y_noisy = librosa.istft(S_noisy * phase)

# Plot isolated noise
plot_audio(y_noisy, sr, "Isolated Noise", 312)

# Plot cleaned audio
plot_audio(y_clean, sr, "Cleaned Audio", 313)

plt.tight_layout()
plt.show()
