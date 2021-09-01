# smoothingASR
Randomized smoothing adversarial defense for ASR models, with original enhancement and voting strategies to limit the drop in performance. This code goes along with the EMNLP 2022 article "Sequential Randomized Smoothing for Adversarially Robust Speech Recognition".

# Setup
We use [armory](https://github.com/twosixlabs/armory) to run adversarial experiments in a controlled environment. Please refer to their README for setup.

To install all requirements we use a custom Docker image. You can set it up with command 
`docker build -t smoothing/pytorch-asr --build-arg armory_version=0.13.4 docker/`

The tag is important, as it is refered to in our config files.

If you do not wish to use docker or armory, please refer to the Dockerfile for all package and libraries installation commands.

# Run
Using armory you can run any of our config file. For instance : 
`armory run configs/pgd/10/g1_trained_rover.json --num-eval-batches 100`

Or write your own config files for custom experiments.

# Models

You will find all our pretrained models and some auxiliary files [here](https://drive.google.com/drive/folders/1MOx0H0Qf_f21pIrkoCPW6VNQJZk3BNe6?usp=sharing). Dump them in the saved models folder you setup when configuring armory.

# Cite
TBD
