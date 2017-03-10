# DeepLearning Toy Models
Toy Models for different kind of Neural Networks

  * CNN: 3 layer network, MNIST data
  * RNN: 8 layer network, 8 bit binary digits
  
## CNN
This is a simple 3 layer Convolutional Neural Network trained on MNIST data. The model is implemented in Matlab without any library dependencies. However, if you wish to check for the correctness of the model, you can also run an identical Python-Tensorflow implementation. Both models are fixed by the same number generator. The purpose of this toy model is that you will be able to see all variables and calculations at a glance in the MATLAB workspace. By intent there is only limited modularisation of functions. So you should be able to see easily what is going on in the training loop and test loop. Forward pass and backward pass are all programmed into one ```CNNtoymodel.m``` m-file.

The limitation of this model is that is uses only 2 dimensional filterbanks. There is no variety of optimizers like Adam, L2-Weight decay or other CNN advanced tricks like Dropout. However for learning purposes feel free to improve the network structure.
### Run and Usage
Just download the CNN folder and run ```CNNtoymodel.m``` in Matlab.

Alternatively run the ```CNNtoymodel.py``` in Python. Before you need to install the following dependencies:
numpy, tensorflow (I use version 12), random, matplotlib and scipy


Epoch=1, learning rate=0.01, batch size training examples=50, batch size test=2000 should give about 80% training accuracy and 67% test accuracy. As you can see there is lot of room for improvement.

## RNN
The Recurrent Neural Network is an 8 layer network. The purpose of this model is to  learn adding two 8 bit binary digits. The Matlab code is re-implementation of a Python Code by Iamtrask. Like in the example above just run the file and you will able to see all variables and calculations in the Matlab workspace. If you like to dig deeper into the theory of this binary RNN, I recommend you to read this blog entry.

### Run and Usage
Just download the RNN folder and run ```RNNtoymodel.m``` in Matlab.
Alternatively run the ```RNNtoymodel.py``` in Python.

Enjoy Roboball

