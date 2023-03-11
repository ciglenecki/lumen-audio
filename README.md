# 🎸 Lumen Data Science 2023 – Audio Classification

Check the code architecture drawing: https://docs.google.com/drawings/d/1DDG480MVKn_C3fZktl5t6uvWeh57Vx2wgtH9GJYsGAU/edit?usp=sharing

![ ](img/code_arh.png)

## Notes

### Meet 2 (2023-03-11, sub)

Tasks:

- normalization!
  - normalization of the audio in time domain (amplitude). Librosa already does this?
  - spectrogram normalization, same as any image problem normalization
    - pre-caculate mean and std and use it
- [ ] audio files which are NOT instruments
  - reserach audio files which are NOT instruments
    - both background noises and sounds SIMILAR to instruments!
    - download the datasets and write dataset loader for them (@matej)
    - label everything \[0, ..., 0\]
- attempt error analysis by looking where the gradients are large
- create eval script which will caculate ALL metrics for the whole dataset
  - precision, f1, confusion matrix, hardest example, scores per instrument
- check validation results
- [ ] Implement ELECTRA

Matej:

- [ ] compare Mirko's wavelet transform with scipy's native transformation
- [ ] implement argument which accepts list of numbers \[1000, 500, 4\] and will create appropriate deep cnn
  - use module called deep head and pass it as a argument
- [ ] compare Mirko's wavelet to scipy wavelet
  - run experiments in both cases
- [ ] check if AST allows for dynamically long audio sequence (longer spectrogram)
  - make sure to perform a forwardpass
  - easiest: resize the spectrogram
- [ ] check batch size n=8 vs n=1 forward pass speed
  - we want to see if we can split the 8sec audio in 1sec sequences to perform forward pass fast
- [ ] perform validation on Rep's corected dataset to check how many labels are correctly marked in the original dataset
  - check if all instruments are correct
  - check if at least one instrument is correct

Mirko:

- [ ] finish experiments and interpretation of the wavelet transformation
- [ ] implement Fluffy on AST, multi-head
- [ ] reserach the BEATs model and incorporate it to the existing training structure as fast as possible so we get concrete results. BEATs links are down below.
- [ ] think about and reserach what happens with variable sampling rate and how can we avoid issues with time length change

Ivan:

- [ ] implement spectrogram cropping and zero padding instead of resizing
- [ ] implement ResNeXt 50_32x4d

Vinko:

- [ ] research audio augmentations
- [ ] research classical audio features
- [ ] implement SVM model which uses classical audio features for mutlilabel classification
  - [ ] research if SVM can perform multilabel classification or use 11 SVMs

### Meet 1

- audio features in the context of traditional approach => baseline

  - https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

- use smaller CNN (efficient-v2-small, imagenet) for intial reporting

  - what's the spectrogram problem in the context of length variability
  - which augmentations do we use?
  - what are the methods for generating new audio files?

- Monolith (Kiklop) vs multi-head (Fluffy):

  - problem with multi-head: number of heads depends on the number of instruments
    - problem with Kiklop but it's manifseted in number of weights
  - Fluffy problem: class disbalans, what's the appropriate loss function. Will the training be stable?

## Setup

### Python Virtual Environment

Create and populate the [virtual environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system). Simply put, the virtual environment allows you to install Python packages for this project only (which you can easily delete later). This way, we won't clutter your global Python packages.

**Step 1: Execute the following command:**

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && pip install -r requirements-dev.txt
```

**Step 2: Install current directory as a editable Python module:**

```bash
pip install -e .
```

**(optional) Step 3: Activate pre-commit hook**

```
pre-commit install
```

Pre-commit, defined in `.pre-commit-config.yaml` will fix your imports will make sure the code follows Python standards

To remove pre-commit run: `rm -rf .git/hooks`

## 📁 Directory structure

| Directory                 | Description                                         |
| ------------------------- | --------------------------------------------------- |
| [data](data/)             | datasets                                            |
| [docs](docs/)             | documentation                                       |
| [figures](figures/)       | figures                                             |
| [models](models/)         | model checkpoints, model metadata, training reports |
| [references](references/) | research papers and competition guidelines          |
| [src](src/)               | python source code                                  |

## 📋 Notes

General links:

- Audio Deep Learning Made Simple State-of-the-Art Techniques:
  1. https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504
  2. https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505
  3. https://towardsdatascience.com/audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52
- paperswithcode Audio Classification: https://paperswithcode.com/task/audio-classification
- Music and Instrument Classification using Deep Learning Technics: https://cs230.stanford.edu/projects_fall_2019/reports/26225883.pdf
- AUDIO MANIPULATION WITH TORCHAUDIO: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

### 🎵 Datasets

IRMAS dataset https://www.upf.edu/web/mtg/irmas:

- IRMAS Test dataset only contains the information about presence of the instruments. Drums and music genre information is **not present**.
- examples:  6705
- instruments: 11
- duration: 3sec

NSynth: Neural Audio Synthesis https://magenta.tensorflow.org/datasets/nsynth

- examples: 305 979
- instruments: 1006
- A novel WaveNet-style autoencoder model that learns codes that meaningfully represent the space of instrument sounds.

MusicNet:

- examples: 330
- instruments: 11
- duration: song

MedleyDB:

- examples: 122
- instruments: 80

OpenMIC-2018 https://zenodo.org/record/1432913#.W6dPeJNKjOR

- paper: http://ismir2018.ircam.fr/doc/pdfs/248_Paper.pdf
- num examples: 20 000
- instruments: 20
- duration: 10sec

### 💡⚙️ Models and training

Problem: how to encode additional features (drums/no drums, music genre)? We can't create spectrogram out fo those arrays. Maybe simply append one hot encoded values after the spectrogram becomes 1D linear vector?

#### BEATs

Current state-of-the-art model for audio classification on multiple datasets and multiple metrics.

paper: https://arxiv.org/pdf/2212.09058.pdf
github: https://github.com/microsoft/unilm/tree/master/beats
https://paperswithcode.com/sota/audio-classification-on-audioset

#### AST

- github: https://github.com/YuanGongND/ast
- paper: https://arxiv.org/pdf/2104.01778.pdf
- pretrained model: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
- hugging face: https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer

Notes:

- They used 16kHz audio for the pretrained model, so if you want to use the pretrained model, please prepare your data in 16kHz

Idea: introduce multiple MLP (fully conneted layer) heads. Each head will detect a single instrument instead of trying to detect all instruments at once.

- [ ] explore how to implement this in PyTorch efficiently:
  - https://ensemble-pytorch.readthedocs.io/en/latest/
  - https://pytorch.org/functorch/stable/notebooks/ensembling.html

Idea: train on single wav, then later introduce `irmas_combinatorics` dataset which contains multiple wav

#### ➕ Ensamble

Introduce SVM and train it additionally on high level features of spectrogram. For example, one can caculate entropy of a spectrogram for a given timeframe.

### Audio knowledge

#### Harmonic and Percussive Sounds

https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S1_HPS.html

![](img/harmonic_and_percussive_sounds.jpg)
Loosely speaking, a harmonic sound is what we perceive as pitched sound, what makes us hear melodies and chords. The prototype of a **harmonic** sound is the acoustic realization of a sinusoid, which corresponds to a **horizontal line in a spectrogram** representation. The sound of a violin is another typical example of what we consider a harmonic sound. Again, most of the observed structures in the spectrogram are of horizontal nature (even though they are intermingled with noise-like components). On the other hand, a percussive sound is what we perceive as a clash, a knock, a clap, or a click. The sound of a drum stroke or a transient that occurs in the attack phase of a musical tone are further typical examples. The prototype of a **percussive** sound is the acoustic realization of an impulse, which corresponds to a **vertical line in a spectrogram representation**.

### 🔊 Feature extraction

https://pytorch.org/audio/stable/transforms.html
https://pytorch.org/audio/stable/functional.html#feature-extractions

#### Spectrogram

note: in practice, Mel Spectrograms are used instead of classical spectrogram. We have to normazlie spectrograms images just like any other image dataset (mean/std).

![ ](img/spectrogram.png)

https://www.physik.uzh.ch/local/teaching/SPI301/LV-2015-Help/lvanls.chm/STFT_Spectrogram_Core.html#:~:text=frequency%20bins%20specifies%20the%20FFT,The%20default%20is%20512.

Take an audio sequence and peform SFTF (Short-time Fourier transform) to get spectrums in multiple time intervals. The result is a 3D tensor (time, amplitude, spectrum). STFT has a time window size which is defined by a `sampling frequnecy`. It is also defined by a `window type`.

### 🥴 Augmentations

- https://pytorch.org/audio/main/tutorials/audio_feature_augmentation_tutorial.html#specaugment
- https://pytorch.org/audio/stable/transforms.html#augmentations
- https://pytorch.org/audio/stable/generated/torchaudio.sox_effects.effect_names.html#torchaudio.sox_effects.effect_names

#### Audio augmentations

- white noise
- time shift
- amplitude change / normalization

<details open>
<summary>PyTorch Sox effects</summary>

allpass, band, bandpass, bandreject, bass, bend, biquad, chorus, channels, compand, contrast, dcshift, deemph, delay, dither, divide, downsample, earwax, echo, echos, equalizer, fade, fir, firfit, flanger, gain, highpass, hilbert, loudness, lowpass, mcompand, norm, oops, overdrive, pad, phaser, pitch, rate, remix, repeat, reverb, reverse, riaa, silence, sinc, speed, stat, stats, stretch, swap, synth, tempo, treble, tremolo, trim, upsample, vad, vol

</details>

#### Spectrum augmentations

SpecAugment: https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html
SpecAugment PyTorch: https://github.com/zcaceres/spec_augment
SpecAugment torchaudio: https://pytorch.org/audio/main/tutorials/audio_feature_augmentation_tutorial.html#specaugment

### 🔀 Data generation

Naive: concat multiple audio sequences into one and merge their labels. Introduce some overlapping, but not too much!

Use the same genre for data generation: combine sounds which come from the same genre instead of different genres

How to sample?

- sample audio files \[3, 5\] but dont use more than 4 instruments
- sample different starting positions at which the audio will start playing
  - START-----x---x----------x--------x----------END
- cutoff the audio sequence at max length?

## 🏆 Team members

<table>
  <tr>
    <td align="center"><a href="https://github.com/VinkoGitHub"><img src="https://avatars.githubusercontent.com/u/32898681?v=4" width="100px;" alt=""/><br /><sub><b>Vinko Dragušica</b></sub><br /></td>
    <td align="center"><a href="https://github.com/mirxonius"><img src="https://avatars.githubusercontent.com/u/102659128?v=4" width="100px;" alt=""/><br /><sub><b>Filip Mirković</b></sub></a><br /></td>
   <td align="center"><a href="https://github.com/ir2718"><img src="https://avatars.githubusercontent.com/u/94498051?v=4" width="100px;" alt=""/><br /><sub><b>Ivan Rep</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/ciglenecki"><img src="https://avatars.githubusercontent.com/u/12819849?v=4" width="100px;" alt=""/><br /><sub><b>Matej Ciglenečki</b></sub></a><br /></td>
</table>
