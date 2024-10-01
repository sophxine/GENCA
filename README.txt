# GENCA: Generative Neural Cellular Automata for Video

## Introduction
![Lenia](https://github.com/sophxine/GENCA/blob/main/lenia.gif)
![Snake](https://github.com/sophxine/GENCA/blob/main/snake.gif)

GENCA uses a novel architecture; a Convolutional Neural Cellular Automata model for predicting next frames based on a sequence of images. It leverages CUDA for efficient training and uses an architecture combining NCA and convolutional layers and optional ODE integration.


**Key Features:**

- **Learns a Cellular Automata Rule:** GENCA learns a set of convolutional filters that act as a cellular automata rule, enabling it to generate new frames based on the spatial patterns in the previous frame.
- **Generalizes and Predicts Future Frames:** GENCA generalizes well to complex systems, it can generalize to unseen initial states, predict future frames beyond the training data and is robust to noise.
- **Transfer Learning:** You can transfer learn across resolutions and datasets, allowing you to leverage knowledge learned to upscale and train on a new dataset.
- **Interactive Influence:** The color drawing feature in the Pygame visualization allows you to directly influence what the NCA generates, determined by which color(state) you draw with. 

To install it, first install the dependecies in requirements.txt.

## Dataset Format

- The dataset should consist of a sequence of images stored in a folder specified by the `image_folder` variable in the code.
- You can use `convert.py` to convert videos and GIFs into image sequences or use your own image sequences, ordered alphanumerically. Clean and organize the images as needed and move them to the training folder. 
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
- **Space:** Pause visualization and training
- **Left mouse button:** Changes cell's color (the default color is white).
- **Right mouse button:** Change draw color based on the color of the cell the mouse is currently hovering.
- **L:** Starts/stops looping prediction for the length of the dataset, initializing with the first frame of the dataset each time. 

## How It Works

GENCA takes a sequence of images as input and learns to predict the next frame in the sequence based on the previous frame. It does this by learning a set of convolutional filters that act as a cellular automata rule. When these filters are applied repeatedly to an initial frame, they generate a sequence of frames that not only approximate the original video but also capture the underlying dynamics of the system, allowing for generalization and future frame prediction. The model also incorporates a state representation, which can potentially improve its ability to remember past information and generate more coherent video sequences.

