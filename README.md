# DeepLearning
Toy Models for different kind of Neural Networks: CNN, RNN.
## CNN
This is a simple 3 layer Convolutional Neural Network trained with MNIST data. The model is implemented in Matlab without any library dependencies. However if you wish to check for the correctness of the model, you can also run an identical Python (Tensorflow) implementation. Both models are fixed by the same number generator. The purpose of this toy model is that you will be able to see all variables and calculations at a glance in the MATLAB workspace. By intent there is only limited modularisation of functions. So you should be able to see easily what is going on in the training loop and test loop. Forward pass and backward pass are all programmed into one file.

The limitation of this model is that is uses only 2 dimensional filterbanks. There is no variety of optimizers like Adam, L2-Weight decay or other CNN tricks like Dropout. However for learning purposes feel free to improve the network structure.


## RNN
The Recurrent Neural Network is an 8 layer network to add two 8 bit binary digits. The Matlab code is re-implementation of a Python Code. Like in the example above just run the file and you will able to see all variables and calculations in the Matlab workspace.

