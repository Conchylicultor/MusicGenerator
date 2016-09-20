# MusicGenerator

## Presentation

Experiment diverse Deep learning models for music generation with TensorFlow

## Results

The different models and experiments are explained [here](docs/models.md).

## Installation

The program requires the following dependencies (easy to install using pip):
 * Python 3
 * TensorFlow (tested with v0.10.0rc0. Won't work with previous versions)
 * CUDA (for using gpu, see TensorFlow [installation page](https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux) for more details)
 * Numpy (should be installed with TensorFlow)
 * Mido (midi library)
 * Tqdm (for the nice progression bars)
 * OpenCv (Sorry, there is no simple way to install it with python 3. It's primarily used as visualisation tool to print the piano roll so is quite optional. All OpenCv calls are contained inside the imgconnector file so if you want to use test the program without OpenCv, you can try removing the functions inside the file)

## Running

To train the model, simply run `main.py`. Once trained, you can generate the results with `main.py --test --sample_length 500`. For more help and options, use `python main.py -h`.

To visualize the computational graph and the cost with TensorBoard, run `tensorboard --logdir save/`.
