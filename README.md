# Compiler classifier based on Graph Convolution NN

[Arxiv paper](https://arxiv.org/abs/1609.02907)

## Data

Compiled functions presented as CFG(Control-flow graph) with GCC and Clang comiler. 

## Problem

Predict predict if compiler was gcc or clang. 

## Model

Simple model with Graph Convolution Layers. (See: `model/gcn.py`) 

## Training

Training lib: Catalyst

Loss: Binary Cross-Entropy

Optimizer: Adam with ReduceLROnPlateau scheduler. 

 

[Control-flow graph](https://en.wikipedia.org/wiki/Control-flow_graph#targetText=In%20computer%20science%2C%20a%20control,matrices%20for%20flow%20analysis%20before.)

