import sounddevice as sd
import scipy.io.wavfile as wav

print("sd.query_devices: ",sd.query_devices())
print("sd.default.devices: ",sd.default.device)
# Record audio in real-time
def record_audio(filename = "sample.wav", duration=15, sample_rate=16000):
    print(f"Recording for {duration} seconds. Please answer the question...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    wav.write(filename, sample_rate, recording)
    print(f"Recording completed and saved as {filename}.")

record_audio()