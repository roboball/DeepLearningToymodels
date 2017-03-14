## Files explained

**Only the 5 files below are needed to run ```CNNtoymodel.m```**

```not_needed``` just contains other functions like tanh etc.  
I just added them as a starting point to improve the model (without actually testing them).  

### Forward Pass
1.```maxpool.m```  is max pooling (downsampling)  
2.```relu.m``` is the activation function 
### Backward Pass
3.```maxpoolup.m``` is the reverse max pooling operation (upsampling)  
4.```reluup.m``` is the reverse relu operation

### Classification
5.```softmax.m``` calculates the Softmax Probabilities, the Cross Entropy Loss, and the Output Delta
