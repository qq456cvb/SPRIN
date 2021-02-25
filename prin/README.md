## Dependencies
* [hydra](https://hydra.cc/) is required for configuration.
* [s2cnn](https://github.com/qq456cvb/s2cnn) with our modification is required to conduct *S2H convolution*. You can run ``git submodule update --init --recursive``, and then run ``python setup.py install`` under ``s2cnn`` folder.
* C++ module on *Density Aware Adaptive Sampling* is required. You can install it under ``prin/src`` by following standard CMake instructions.

## Train/Test
Once the dataset is placed under the root directory, just run
``
python sprin/train.py
``
for training, and 
``
python sprin/train.py
``
for testing.