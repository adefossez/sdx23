# This file uses Hybrid Demucs for music demixing.
# It is one of official baselines for the Music Demixing challenge.
#
# Reference: Alexandre Défossez. "Hybrid Spectrogram and Waveform Source Separation"
#            MDX Workshop at ISMIR 2021
#
# NOTE:
# a) Demucs needs checkpoints to be submitted along with your code.
# b) Please upgrade Demucs to the latest release (4.0.0).
#
# If you trained your model with the Demucs codebase, make sure to export
# your model, using e.g. `python -m tools.export SIG`. Then copy the files
# `release_models/SIG.th` into this repo.
# Update the SIG in the get_model hereafter.
#
# /!\ Remember to update the aicrowd.json to match your use case.
#
# Making submission using demucs:
# 2. Run this file locally with `python evaluate_locally.py`.
# 4. Submit your code using git-lfs
#    #> git lfs install
#    #> git lfs track "*.th"
#    #> git add .gitattributes
#    #> git add models
#    #> git add -u .

# Follow the instructions in the docs/submission.md file.
# Once the repo is properly setup, you can easily push new submissions with
# > git add models; git add -u .
# > git commit -m "commit message"
# > name="submission name here" ; git tag -am "submission-$name" submission-$name; git push aicrowd submission-$name


from pathlib import Path
import time
import numpy as np
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio
import torch


class Demucs:
    """
    Demucs model for music demixing.
    """
    def __init__(self):
        # 053b5e8c is for labelnoise
        # 17a36a86 is for bleeding
        # If you participate in both competitions do not forget to update aicrowd.json 
        # with the proper task !!!
        self.separator = pretrained.get_model('17a36a86', repo=Path('./models'))

        # we select automatically the device based on what is available,
        # remember to change in aicrowd.json if you want to use the GPU or not.
        if torch.cuda.device_count():
            self.device = torch.device('cuda')
        else:
            print("WARNING, using CPU")
            self.device = torch.device('cpu')
        self.separator.to(self.device)

    @property
    def instruments(self):
        """ DO NOT CHANGE """
        return ['bass', 'drums', 'other', 'vocals']

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)

    def separate_music_file(self, mixed_sound_array, sample_rate):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """

        # create input for Demucs model
        mix = torch.from_numpy(np.asarray(mixed_sound_array.T, np.float32))

        # convert audio to GPU
        mix = mix.to(self.device)
        mix_channels = mix.shape[0]
        mix = convert_audio(mix, sample_rate, self.separator.samplerate, self.separator.audio_channels)

        b = time.time()
        # Normalize track, no required for any recent version of Demucs but never hurts.
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std
        # Separate
        with torch.no_grad():
            estimates = apply_model(self.separator, mix[None], overlap=0.15, progress=False)[0]

        sr = self.separator.samplerate
        # Printing some sanity checks.
        print(time.time() - b, mono.shape[-1] / sr, mix.std(), estimates.std())

        estimates = estimates * std + mean

        estimates = convert_audio(estimates, self.separator.samplerate, sample_rate, mix_channels)

        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            idx = self.separator.sources.index(instrument)
            separated_music_arrays[instrument] = torch.squeeze(estimates[idx]).detach().cpu().numpy().T
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates
