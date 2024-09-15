import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, filtfilt

# Load the audio file
y, sr = librosa.load("/Users/shaanmehta/Downloads/noisy.wav", sr=None)

# Perform Short-Time Fourier Transform (STFT)
S_full, phase = librosa.magphase(librosa.stft(y))

# Define high-pass filter cutoff frequency
cutoff = 5000  # Hz, adjust as needed
frequencies = np.fft.fftfreq(S_full.shape[0], d=1/sr)
high_pass_filter = np.abs(frequencies) > cutoff

# Apply the filter
S_filtered = S_full * high_pass_filter[:, None]

# Reconstruct the time-domain signal from the filtered magnitude and original phase
y_filtered = librosa.istft(S_filtered * phase)

# Write the filtered audio to a file
sf.write("/Users/shaanmehta/Downloads/high_pass_filtered.wav", y_filtered, sr)

# Plotting the original, filtered audio
def plot_audio(signal, sr, title, subplot):
    plt.subplot(subplot)
    plt.plot(np.linspace(0, len(signal) / sr, num=len(signal)), signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.figure(figsize=(14, 10))

plot_audio(y, sr, "Original Noisy Audio", 311)

# Plot high-pass filtered audio
plot_audio(y_filtered, sr, "High-Pass Filtered Audio", 312)

plt.tight_layout()
plt.show()
