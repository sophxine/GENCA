GENCA is a Neural Cellular Automata for generating videos and predicting future frames.

It supports CUDA so make sure you have CUDA installed to efficiently train it.

The dataset format should be images in the folder defined in image_folder.
You can use convert.py to convert videos and gifs to image sequences and after cleaning it if you want to, move the images to the training folder. The data contains an example with Lenia.
In visualization it loads the first image of the dataset.
To run a saved model without loading the dataset(excluding the first for initialization) or training, set the parameters to this:
train = False
visualize = True


Model architecture:

It's a Neural Cellular Automata using 2D convolutional layers with state representation and ODE(ordinary differential equations) integration.


Summary of how it works:

It takes in a sequence of images, and outputs next frame from previous frame.
In the training loop it takes a frame of the dataset and predicts the next frame, for all frames of the dataset
It learns convolution filters that when applied iteratively, just like a cellular automata rule, to the initial frame in the real sequence, result in an approximation of the sequence and it's future frames.
It also has state for potentially better memory but I have not tested that.

You can transfer learn across resolutions and datasets.


Keys available in the pygame visualization:

S: Save model
I: initialize with the first frame of the dataset
Tab: Randomize grid
Space: Pause
Left mouse button: Changes cell's color(the default color is white).
Right mouse button: Change draw color based on color of the cell the mouse is currently hovering.
