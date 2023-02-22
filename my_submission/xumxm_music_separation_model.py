# This file uses the X-version of Open Unmix (XUMX-M) for music demixing.
# It is one of official baselines for the Music Demixing challenge.
#
# Reference: Sawata, Ryosuke, et al. "All for one and one for all: Improving music separation by bridging networks."
#            ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.
#
# NOTE:
# a) XUMX-M needs checkpoint to be submitted along with your code.
#
# Making submission using openunmix:
# 1. Change the model in `user_config.py` to XUMXMMusicSeparationModel.
# 2. Run this file locally with `python evaluate_locally.py`.
# 3. Set `external_dataset_used` to `true` in your `aicrowd.json`.
# 4. Submit your code using git-lfs
#    #> git lfs install
#    #> git lfs track "*.pth"
#    #> git add .gitattributes
#    #> git add models
from asteroid.models import XUMX
from asteroid.complex_nn import torch_complex_from_magphase
import norbert
import numpy as np
import os.path
import scipy
import torch


# Inverse STFT - taken from
#    https://github.com/asteroid-team/asteroid/blob/master/egs/musdb18/X-UMX/eval.py
def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2), rate, nperseg=n_fft, noverlap=n_fft - n_hopsize, boundary=True
    )
    return audio

# Separation function - taken from
#    https://github.com/asteroid-team/asteroid/blob/master/egs/musdb18/X-UMX/eval.py
def separate(
    audio,
    x_umx_target,
    instruments,
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    device="cpu",
):
    """
    Performing the separation on audio input
    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio
    x_umx_target: asteroid.models
        X-UMX model used for separating
    instruments: list
        The list of instruments, e.g., ["bass", "drums", "vocals"]
    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.
    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False
    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0
    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False
    device: str
        set torch device. Defaults to `cpu`.
    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary with all estimates obtained by the separation model.
    """

    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    # put model on correct device
    x_umx_target.to(device)

    source_names = []
    V = []

    masked_tf_rep, _ = x_umx_target(audio_torch)
    # shape: (Sources, frames, batch, channels, fbin)

    for j, target in enumerate(instruments):
        Vj = masked_tf_rep[j, Ellipsis].cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj ** alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, Ellipsis])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    # convert to complex numpy type
    tmp = x_umx_target.encoder(audio_torch)
    X = torch_complex_from_magphase(tmp[0].permute(1, 2, 3, 0), tmp[1])
    X = X.detach().cpu().numpy()
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(instruments) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += ["residual"] if len(instruments) > 1 else ["accompaniment"]

    Y = norbert.wiener(V, X.astype(np.complex128), niter, use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            rate=x_umx_target.sample_rate,
            n_fft=x_umx_target.in_chan,
            n_hopsize=x_umx_target.n_hop,
        )
        estimates[name] = audio_hat.T

    return estimates


class XUMXMMusicSeparationModel:
    """
    X-UMXM separation model.
    """
    def __init__(self):
        # check if model was already downloaded
        if not os.path.isfile('./models/pretrained_xumxm.pth'):
            import urllib.request
            urllib.request.urlretrieve('https://zenodo.org/record/7128659/files/pretrained_xumxl.pth?download=1', './models/pretrained_xumxm.pth')

        # load model
        self.separator = XUMX.from_pretrained("./models/pretrained_xumxm.pth")
        self.separator.eval()

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

        # pad mixture to compensate STFT truncation
        mixed_sound_array_padded = np.pad(mixed_sound_array, ((0, 1024), (0, 0)))

        # perform separation
        estimates = separate(
            mixed_sound_array_padded,
            self.separator,
            self.separator.sources,
            device='cuda',
        )

        # truncate to orignal length
        for target in estimates:
            estimates[target] = estimates[target][:mixed_sound_array.shape[0], :]

        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            separated_music_arrays[instrument] = estimates[instrument]
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates
