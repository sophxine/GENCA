# GENCA: Generative Neural Cellular Automata for Video

## Introduction

GENCA is a Neural Cellular Automata (NCA) model designed for video generation and future frame prediction. It leverages the power of CUDA for efficient training and utilizes a novel architecture combining convolutional layers, state representation, and ODE integration. 

**Key Features:**

- **Learns a Cellular Automata Rule:** GENCA learns a set of convolutional filters that act as a cellular automata rule, enabling it to generate new frames based on the spatial patterns in the previous frame.
- **Generalizes and Predicts Future Frames:** GENCA goes beyond simply approximating the training video. It learns the underlying rules of the system, allowing it to generalize to unseen initial states and predict future frames beyond the training data.
- **Transfer Learning:** You can transfer learn across resolutions and datasets, allowing you to leverage knowledge learned to upscale and train on a new dataset.
- **Interactive Influence:** The color drawing feature in the Pygame visualization allows you to directly influence what the NCA generates. The model learns to respond to different color patterns. For example, in a model trained on a video of a flowing river, drawing blue might cause the NCA to generate more water in that spot, while drawing green might encourage the growth of vegetation. 

## Requirements

- CUDA-enabled GPU and CUDA toolkit installed.
- PyTorch
- Pygame
- Other necessary Python libraries (use pip install -r requirements.txt).

## Dataset Format

- The dataset should consist of a sequence of images stored in a folder specified by the `image_folder` variable in the code.
- You can use the provided `convert.py` script to convert videos and GIFs into image sequences or use your own image sequences, ordered alphanumerically. 
- Clean and organize the images as needed and move them to the training folder. 
- The "data" folder contains an example dataset with Lenia.

## Running a Saved Model

To run a saved model without loading the dataset (except the first frame for initialization) or training, set the following parameters:

`train = False`
`visualize = True`

**Note:** In the pygame visualization, it loads and initializes with the first image of the dataset.

## Pygame Controls

- **S:** Save model
- **I:** Initialize with the first frame of the dataset
- **Tab:** Randomize grid
- **Space:** Pause
- **Left mouse button:** Changes cell's color (the default color is white).
- **Right mouse button:** Change draw color based on the color of the cell the mouse is currently hovering.

## How It Works (Simplified)

GENCA takes a sequence of images as input and learns to predict the next frame in the sequence based on the previous frame. It does this by learning a set of convolutional filters that act as a cellular automata rule. When these filters are applied repeatedly to an initial frame, they generate a sequence of frames that not only approximate the original video but also capture the underlying dynamics of the system, allowing for generalization and future frame prediction. The model also incorporates a state representation, which can potentially improve its ability to remember past information and generate more coherent video sequences.
