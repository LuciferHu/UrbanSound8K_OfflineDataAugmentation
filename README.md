# UrbanSound8K_OfflineDataAugmentation
Use a method of  mixing an audio file with a noise file at any Signal-to-Noise Ratio

This project based on the repository [audio-SNR](https://github.com/Sato-Kunihiko/audio-SNR) and was built to mix examples from Dataset [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) with a noise file from [DCASE2020 Task1](http://dcase.community/challenge2020/task-acoustic-scene-classification).

# Denpencies
librosa  0.8.0
pysoundfile  0.10.3.post1

# File format
To mix audios, we should know the format of the audio files. Waveforms form UrbanSound8K have many different formats, which are listed as follows:

- PCM16, 2 channels, 44100Hz, WAV
- PCM16, 2 channels, 48000Hz, WAV
- PCM16, 2 channels, 22000Hz, WAV
- PCM24, 2channels, 44100Hz, WAVX
- PCM24, 2channels, 48000Hz, WAVX

note: WAVX is from microsoft and cannot be processed with built-in module [wave](https://docs.python.org/3/library/wave.html).

 Waveforms form DCASE2020 have just one kind of format:
 
 -PCM24, 1 channel, 44100Hz, WAV
 Note that I have reformat the audios to PCM16.
 
 # Usage
 There are three files for creating a mixture.
 - 'audio_mix':
      - Use librosa to load wav audo and normalize data to one channel of float type data.
      - Use pysoundfile to write mixture audio with the same subtype of urbansound audio.
     
 - 'make_dataframe':
      - load csvs. Note that I choose 7 classes of UrbanSound8k and add another 2 class of data which are built from my own. Only 4 street scenes, which are park, public_square, street_pedestrian and street_traffic, are chose to be the background noise. 
     
 - 'offline_data_aumentation':
      - data mixing. All classes are augmented to 2000 examples. Some classes has fewer raw examples, so I sampled more to synthesize enough example. For example, class 'car_horn' has 429 raw examples, and I sampled 329. Other classes has enough examples and was sampled 1/4 of the original.
