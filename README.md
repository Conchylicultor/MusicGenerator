# MusicGenerator

## Presentation

Experiment diverse Deep learning models for music generation with TensorFlow

## Installation

The program require the following dependencies (easy to install using pip):
 * python 3
 * tensorflow (tested with v0.9.0)
 * numpy
 * CUDA (for using gpu, see TensorFlow [installation page](https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux) for more details)
 * mido (midi library)
 * tqdm (for the nice progression bars)

## Running

To train the model, simply run `main.py`. Once trained, you can generate the results with `main.py --test`. For more help and options, use `python main.py -h`.

To visualize the computational graph and the cost with TensorBoard, just run tensorboard --logdir save/. .

## Results

...
