# Deep Learning Toy Models
Toy Models for different kind of Neural Networks  
  * CNN: 3 layer network, MNIST data  
  * RNN: 8 layer network, 8 bit binary digits
  
## CNN
This is a simple 3 layer Convolutional Neural Network trained on MNIST data. Forward and backward pass are all programmed into one ```CNNtoymodel.m``` file (no libraries are needed). By intent there is only limited modularisation of functions. So it should be easy for you to see what is going on in the training loop and test loop by a glance at the MATLAB workspace. However, if you wish to check for the correctness of the model, you can also run an identical Tensorflow implementation by excecuting the ```CNNtoymodel.py``` in Python. 
### Run and Usage
- **Matlab**: Just download the CNN folder and run ```CNNtoymodel.m``` in Matlab.   
- **Python**: Alternatively run the ```CNNtoymodel.py``` in Python.  

For python you need to install the following dependencies:  
```numpy, tensorflow (I use version 12), random, matplotlib and scipy```

One epoch, learning rate=0.01, batch size training=50, batch size test=2000 should give about 80% training accuracy and 67% test accuracy. Limitations of this model are that it uses only 2D filterbanks instead of 4D volumes. Further the model does not use any advanced CNN optimization techniques like Momentum or Dropout. Feel free to improve the model.

## RNN
The Recurrent Neural Network is an 8 layer network. The purpose of this model is to learn adding two 8 bit binary digits. The Matlab code is re-implementation of the Python Code by Iamtrask. Like in the example above just run the file and you will able to see all variables and calculations in the Matlab workspace. If you like to dig deeper into the theory of this binary RNN, I recommend you to read this blog about [binary RNN background info](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/).

### Run and Usage
- **Matlab**: Just download the RNN folder and run ```RNNtoymodel.m``` in Matlab.  
- **Python**: Alternatively run the ```RNNtoymodel.py``` in Python.


Enjoy Roboball

