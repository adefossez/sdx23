# This file uses Open Unmix Extra (UMX-L) for music demixing.
# It is one of official baselines for the Music Demixing challenge.
#
# Reference: Stöter, Fabian-Robert, et al. "Open-unmix-a reference implementation for music source separation."
#            Journal of Open Source Software 4.41 (2019): 1667.
#
# NOTE:
# a) Open Unmix Extra (UMX-L) needs checkpoints to be submitted along with your code.
# b) Please upgrade to openunmix>=1.2.0 for UMX-L.
#
# Making submission using openunmix:
# 1. Change the model in `user_config.py` to UMXLMusicSeparationModel.
# 2. Run this file locally with `python evaluate_locally.py`.
# 3. Set `external_dataset_used` to `true` in your `aicrowd.json`.
# 4. Submit your code using git-lfs
#    #> git lfs install
#    #> git lfs track "*.pth"
#    #> git add .gitattributes
#    #> git add models

import numpy as np
from openunmix import predict
import torch

class UMXLMusicSeparationModel:
    """
    UMX-L model for music demixing.
    """
    def __init__(self):
        # Load UMX model
        torch.hub.set_dir('./models')
        """
        try:
            # try to load model from cache
            self.separator = torch.hub.load("./models/sigsep_open-unmix-pytorch_master", "umxl", source='local')
        except:
            # could not find cached version - download from PyTorch hub
            self.separator = torch.hub.load("sigsep/open-unmix-pytorch", "umxl")
        """
        self.separator = torch.hub.load("./models/sigsep_open-unmix-pytorch_master", "umxl", source='local')

        # copy to GPU
        self.device = torch.device('cuda')
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

        # create input for UMXL model
        mixed_sound_tensor = torch.from_numpy(np.asarray(mixed_sound_array.T, np.float32))

        # convert audio to GPU
        mixed_sound_tensor = mixed_sound_tensor.to(self.device)

        # run UMX model
        with torch.inference_mode():
           estimates = predict.separate(
                audio=mixed_sound_tensor,
                rate=sample_rate,
                separator=self.separator,
                niter=1
            )

        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            separated_music_arrays[instrument] = torch.squeeze(estimates[instrument]).detach().cpu().numpy().T
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates
