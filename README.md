# Acoustic neural distance demo

## Overview
This project is a demo for the computation of acoustic neural distances between **single word** sound recordings. The algorithm is intended to compare sound files that contain similar utterances, sucuh as two dialectal variants of a certain target word. We do this here specifically for the  Wav2Vec2-XLSR-53 model, which has been finetuned for speech data from the Netherlands (see the [Hugging Face model card](https://huggingface.co/GroNLP/wav2vec2-dutch-large-ft-cgn)). Other wav2vec 2.0 models can be used as well, for example for different geographical areas.

## Getting started
### Preparing audio files
In order to run the computation, ensure that the sound files that you want to compare are in separate folders in the recordings folder. For the example data, we put the recordings in the folders 'set1' and 'set2'. The sound files should be 16 kHz PCM WAV files and contain no silences before or after the single word. This can be achieved by removing silences using sox or Voice Activity Detected methods (such as [Pyannote](https://github.com/pyannote/pyannote-audio)). 


## Preparing the environment
Ensure that you have `poetry` [(see here)](https://python-poetry.org/docs/#installing-with-the-official-installer) installed.

Go into the main directory with your command line program. To get started and install the required dependencies, run the following:

```
poetry update
```


<!-- ### Downloading acoustic model
Go into the directory containing the recordings folder (i.e., *demo_acoustic_neural_distance*) with your command line program. Then run:

```
cd demo_acoustic_neural_distance
git clone git@hf.co:GroNLP/wav2vec2-dutch-large-ft-cgn
```
 -->

## Computing differences
To compute the distances, run (ensuring that you are in the correct folder).:

```
poetry run python compute_differences.py
```

The distances are consequently saved in distances.txt.

If you run into any problems, ensure that you are in the right folder to run the code: *demo-acoustic-neural-distance*. 