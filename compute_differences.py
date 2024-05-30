from itertools import combinations, permutations
from pathlib import Path
import json
import multiprocessing
import pickle

from dtw import dtw
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from praatio import textgrid
import torch
from transformers import logging
from transformers.models.wav2vec2 import Wav2Vec2Model
from tqdm import tqdm

logging.set_verbosity_error()


@torch.no_grad()
def _featurize(audio, srate, layer, featurizer):
    layer = int(layer)

    # flatten input values and return as tensor
    input_values = torch.from_numpy(audio).unsqueeze(0)

    # if GPU is available, move input values to GPU
    if torch.cuda.is_available():
        input_values = input_values.cuda()

    # obtain hidden states
    if layer is None:
        hidden_states = featurizer(
            input_values, output_hidden_states=True
        ).hidden_states
        hidden_states = [s.squeeze(0).cpu().numpy() for s in hidden_states]
        return hidden_states

    if layer >= 0:
        hidden_state = (
            featurizer(input_values).last_hidden_state.squeeze(0).cpu().numpy()
        )
    else:
        hidden_state = featurizer.feature_extractor(input_values)
        hidden_state = hidden_state.transpose(1, 2)
        if layer == -1:
            hidden_state = featurizer.feature_projection(hidden_state)
        hidden_state = hidden_state.squeeze(0).cpu().numpy()

    return hidden_state


def load_wav2vec2_featurizer(model_name_or_path, layer=None):
    """
    Loads Wav2Vec2 featurization pipeline and returns it as a function.

    Featurizer returns a list with all hidden layer representations if "layer" argument is None.
    Otherwise, only returns the specified layer representations.
    """

    model_kwargs = {}
    if layer is not None:
        model_kwargs["num_hidden_layers"] = layer if layer > 0 else 0
    model = Wav2Vec2Model.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    return model


def dtw_distance(a_feats, b_feats):
    if a_feats.shape[0] < 2 or b_feats.shape[0] < 2:
        return np.nan

    dist = dtw(
        a_feats,
        b_feats,
        distance_only=True,
    ).normalizedDistance

    return dist


if __name__ == "__main__":
    # choose model, layer, and initialize featurizer
    # model = "wav2vec2-large-xlsr-53-ft-cgn"
    model = "mms-300m"
    layer = "15"
    featurizer = load_wav2vec2_featurizer(model, int(layer))

    files = [f for f in Path("recordings").rglob("**/*.wav") if f.is_file()]
    print(files)

    # load audio files and featurize them
    audio_features = {}
    for file in files:
        audio, srate = librosa.load(file, sr=16000)
        audio_features[file] = _featurize(audio, srate, layer, featurizer)

    # compute DTW distances
    distances = {}
    for a, b in combinations(audio_features.keys(), 2):
        distances[(a, b)] = dtw_distance(audio_features[a], audio_features[b])

    # write to file

    with open("distances.txt", "w") as f:
        for (a, b), dist in distances.items():
            f.write(f"{a}\t{b}\t{dist}\n")