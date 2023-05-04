import numpy as np
import soundfile as sf

rate = 16_000
seconds = [0.01, 0.1, 1, 5, 10, 60, 60 * 5, 60 * 30, 60 * 60]
audios = [np.random.rand(1, int(s * rate)) for s in seconds]

for a, s in zip(audios, seconds):
    # sf.write(f"test.wav", a, rate, format="wav", subtype="PCM_24")
    sf.write(
        f"sec_{str(s).replace('.', '_')}.wav", a.T, rate, format="wav", subtype="PCM_24"
    )
