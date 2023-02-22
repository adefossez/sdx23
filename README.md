![Banner image](https://images.aicrowd.com/uploads/ckeditor/pictures/1040/content_Desktop_Banner.png)

# **[Music Demixing Challenge 2023 - Music Separation](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23)** - Starter Kit
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the Sound Demixing Challenge 2023 - Music Separation **Starter kit**! It contains:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!

Quick Links:

* [Sound Demixing Challenge 2023 - Music Separation Track - Competition Page](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23)
* [Discussion Forum](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/discussion)
* [Sound Demixing 2023 Challenge Overview](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/)


# 📝 Table of Contents
1. [About the Sound Demixing Challenge 2023](#about-the-sound-demixing-challenge-2023)
2. [Evaluation](#evaluation)
3. [Baselines](#baselines) 
4. [How to test and debug locally](#how-to-test-and-debug-locally)
5. [How to submit](#how-to-submit)
6. [Dataset](#dataset)
7. [Setting up your codebase](#setting-up-your-codebase)
8. [FAQs](#faqs)

# 🎶 About the Sound Demixing Challenge 2023

Have you ever sung using a karaoke machine or made a DJ music mix of your favourite song? Have you wondered how hearing aids help people listen more clearly or how video conference software reduces background noise? 

They all use the magic of audio separation. 

Music source separation (MSS) attracts professional music creators as it enables remixing and revising songs in a way traditional equalisers don't. Suppressed vocals in songs can improve your karaoke night and provide a richer audio experience than conventional applications. 

The Sound Demixing Challenge 2023 (SDX23) is an opportunity for researchers and machine learning enthusiasts to test their skills by creating a system to **perform audio source separation**. 

Given an **audio signal as input** (referred to as a "mixture"), you must **decompose in its different parts**. 

![separation image](https://images.aicrowd.com/uploads/ckeditor/pictures/401/content_image.png)

## 🎻 MUSIC SEPARATION

This task will focus on music source separation. Participants will submit systems that separate a song into four instruments: vocals, bass, drums, and other (the instrument "other" contains signals of all instruments other than the first three, e.g., guitar or piano). 

Karaoke systems can benefit from the audio source separation technology as users can sing over any original song, where the vocals have been suppressed, instead of picking from a set of "cover" songs specifically produced for karaoke.

Similar to [Music Demixing Challenge 2021](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021), this task will have two leaderboards.

### **Leaderboard A (Labelnoise)**

To submit to Leaderboard A, you should only use the Labelnoise dataset for training.

🚨 **NOTE**: To participate in Leaderboard A, please set exactly the following setting in the [`aicrowd.json`](aicrowd.json) file.

```
"labelnoise": true,
"bleeding": false,
"external_dataset_used": false,
```


### **Leaderboard B (Bleeding)**

To submit to Leaderboard A, you should only use the Bleeding dataset for training.

🚨 **NOTE**: To participate in Leaderboard A, please set exactly the following setting in the [`aicrowd.json`](aicrowd.json) file.

```
"labelnoise": false,
"bleeding": true,
"external_dataset_used": false,
```

### **Leaderboard C**

All submissions will get entry for Leaderboard C.
 
# ✅ Evaluation

As an evaluation metric, we are using the signal-to-distortion ratio (SDR), which is defined as,

$`SDR_{instr} = 10log_{10}\frac{\sum_n(s_{instr,left\ channel}(n))^2 + \sum_n(s_{instr,right\ channel}(n))^2}{\sum_n(s_{instr,left\ channel}(n) - \hat{s}_{instr,left\ channel}(n))^2 + \sum_n(s_{instr,right\ channel}(n) - \hat{s}_{instr,right\ channel}(n))^2}`$

where $S_{instr}(n)$ is the waveform of the ground truth and Ŝ𝑖𝑛𝑠𝑡𝑟(𝑛) denotes the waveform of the estimate. The higher the SDR score, the better the output of the system is.

In order to rank systems, we will use the average SDR computed by

$`SDR_{song} = \frac{1}{4}(SDR_{bass} + SDR_{drums} + SDR_{vocals} + SDR_{other})`$

for each song. Finally, the overall score is obtained by averaging SDRsong over all songs in the hidden test set.

# 🤖 Baselines

We use the [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) library for the baseline. Specifically, we provide trained checkpoints for the UMXL model. You can use the baseline by switching to the `openunmix-baseline` [branch](https://gitlab.aicrowd.com/aicrowd/challenges/sound-demixing-challenge-2023/sdx-2023-music-demixing-track-starter-kit/-/blob/openunmix-baseline/) on this repository. To test the models locally, you need to install `git-lfs`.

When submitting your own models, you need to submit the checkpoints using `git-lfs`. Check the instructions shared in the inference file [here](https://gitlab.aicrowd.com/aicrowd/challenges/sound-demixing-challenge-2023/sdx-2023-music-demixing-track-starter-kit/-/blob/openunmix-baseline/my_submission/openunmix_separation_model.py)

# 💻 How to Test and Debug Locally

The best way to test your models is to run your submission locally.

You can do this by simply running  `python evaluate_locally.py`. **Note that your local setup and the server evalution runtime may vary.** Make sure you mention setup your runtime according to the section: [How do I specify my dependencies?](#how-do-i-specify-my-dependencies)

# 🚀 How to Submit

You can use the submission script `source submit.sh <submission_text>`

More information on submissions can be found in [SUBMISSION.md](/docs/submission.md).

#### A high level description of the Challenge Procedure:
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/music-demixing-challenge-2023).
2. **Clone** this repo and start developing your solution.
3. **Train** your models on IGLU, and ensure run.sh will generate rollouts.
4. **Submit** your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com)
for evaluation (full instructions below). The automated evaluation setup
will evaluate the submissions against the IGLU Gridworld environment for a fixed 
number of rollouts to compute and report the metrics on the leaderboard
of the competition.


# 💽 Dataset

Download the public dataset for this task using this [link](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23/dataset_files), you'll need to accept the rules of the competition to access the data. The data is same as the well known MUSDB18-HQ dataset and its compressed version.


# 📑 Setting Up Your Codebase

AIcrowd provides great flexibility in the details of your submission!  
Find the answers to FAQs about submission structure below, followed by 
the guide for setting up this starter kit and linking it to the AIcrowd 
GitLab.

## FAQs

### How do I submit a model?
In short, you should push you code to the AIcrowd's gitlab with a specific git tag and the evaluation will be triggered automatically. More information on submissions can be found at our [submission.md](/docs/submission.md). 

### How do I specify my dependencies?

We accept submissions with custom runtimes, so you can choose your 
favorite! The configuration files typically include `requirements.txt` 
(pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about this in [runtime.md](/docs/runtime.md).

### What should my code structure look like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:


```
.
├── aicrowd.json                # Add any descriptions about your model, set `external_dataset_used`, and gpu flag
├── apt.txt                     # Linux packages to be installed inside docker image
├── requirements.txt            # Python packages to be installed
├── evaluate_locally.py         # Use this to check your model evaluation flow locally
└── my_submission               # Place your models and related code here
    ├── <Your model files>      # Add any models here for easy organization
    ├── aicrowd_wrapper.py      # Keep this file unchanged
    └── user_config.py          # IMPORTANT: Add your model name here
```

### How can I get going with a completely new model?

Train your model as you like, and when you’re ready to submit, implement the inference class and import it to `my_submission/user_config.py`. Refer to [`my_submission/README.md`](my_submission/README.md) for a detailed explanation.

Once you are ready, test your implementation `python evaluate_locally.py`

### How do I actually make a submission?

You can use the submission script `source submit.sh <submission_text>`

The submission is made by adding everything including the model to git,
tagging the submission with a git tag that starts with `submission-`, and
pushing to AIcrowd's GitLab. The rest is done for you!

For large model weight files, you'll need to use `git-lfs`

More details are available at [docs/submission.md](/docs/submission.md).

When you make a submission browse to the `issues` page on your repository, a sucessful submission should look like this.

![submission image](https://images.aicrowd.com/uploads/ckeditor/pictures/1041/content_Screenshot_from_2022-12-01_17-16-12.png)

### How to use GPU?

To use GPU in your submissions, set the gpu flag in `aicrowd.json`. 

```
    "gpu": true,
```

### Are there any hardware or time constraints?

Your submission will need to complete predictions on each **sound tracks** under **1.57 x duration of the track**. Make sure you take advantage of all the cores by parallelizing your code if needed. Incomplete submissions will fail.

The machine where the submission will run will have following specifications:
* 4 vCPUs
* 16GB RAM
* (Optional) 1 NVIDIA T4 GPU with 16 GB VRAM - This needs setting `"gpu": true` in `aicrowd.json`

# 📎 Important links
- 💪 [Challenge Page](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23)
- 🗣️ [Discussion Forum](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/discussion)
- 🏆 [Leaderboard](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23/leaderboards)
- 🎵 [Music Demixing Challenge 2021](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021)

You may also like the new [Cinematic Sound Separation track](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/cinematic-sound-demixing-track-cdx-23)

**Best of Luck** 🎉 🎉
