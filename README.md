
## Sound Generation using Conditional Variational Autoencoders

### Description

Generative Models are playing an important role in current developments in Artificial Intelligence applied to different areas such as Natural Language Processing and Computer Vision. Similarly, the field of generative audio models is constantly evolving, and it shows promising results. This repository contains various experiments to explore the efficacy of Variational Autoencoders (VAEs) and Conditional Variational Autoencoders (CVAEs) in capturing and synthesizing diverse features from recordings of Musical Instruments. By training these models on spectrograms, it is possible to generate high-fidelity sound representations under different conditions.

### Setup Environment

1. Use an Anaconda environment.
2. Download the dataset from IRCAM:
   - [OrchideaSOL dataset](https://forum.ircam.fr/projects/detail/orchideasol/)
   - [TinySOL dataset (Preferred)](https://forum.ircam.fr/projects/detail/tinysol/)

### Usage

1. Start by exploring the dataset using the `data_exploration` notebook. This notebook allows you to review metadata and organize it depending on your needs.
2. Create subsets of musical instruments in CSV files to easily access the audio clips.
3. Use the notebooks in the folders to try different VAEs. All of them look very similar but use a different model.
4. Note that Convolutional VAEs require more time to train.

This repository is being updated periodically.
