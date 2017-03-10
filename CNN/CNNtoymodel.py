#=======================================================================
#Purpose: Classification of Digits with a CNN
#
#Model:   3 Layer Convolutional Neural Network
#         Trained with Minibatch Gradient Descent
#
#Inputs:  MNIST Database (digits)
#         60,000 training set
#         10,000 test set 
#
#Output:  10 Classes (digits from 1 to 10)
#Version: 10/2016 Roboball (MattK.)
#Link:    https://github.com/roboball/DeepLearningToymodels
#=======================================================================

import numpy as np
import tensorflow as tf
import random
import os, re
import datetime 
import matplotlib.pyplot as plt 
import scipy.io
#plt.close('all') 

########################################################################
#import data from mat-file: MNIST
########################################################################

print('init 3 layer CNN')
print('start loading MNIST data\n')
mat = scipy.io.loadmat('data/MNIST.mat')

#convert to numpy arrays:
imagetrainxx = np.array(mat['imagetrain'])
labeltrain = np.array(mat['labeltrain'],dtype=int)
imagetestxx = np.array(mat['imagetest'])
labeltest = np.array(mat['labeltest'],dtype=int)

#-----------------------------------------------------------------------
#convert labels to one hot encoding
#-----------------------------------------------------------------------
labtrainone = np.zeros((60000, 10))
labtestone = np.zeros((10000, 10))

#convert training labels
for one in range(60000):
	labtrainone[one][labeltrain[one]] = 1

#convert test labels
for one in range(10000):
	labtestone[one][labeltest[one]] = 1

########################################################################
#init, define and train: CNN model
########################################################################

#define parameter-------------------------------------------------------
epochs = 1
learnrate=0.01
train_size = 60000
test_size = 10000
batsize_train=50
batsize_test=2000
batches_train = int(train_size / batsize_train)
batches_test = int(test_size / batsize_test)


#init: cost history
cost_history = np.empty(shape=[0],dtype=float)
#init: accuracy
acc_history = np.empty(shape=[0],dtype=float)
#init: batches correct
correctbatch = np.empty(shape=[0],dtype=float)

#init: grads and vars
gradvar_history = np.empty(shape=[0],dtype=float)
#init: grads and vars
applygrad_history = np.empty(shape=[0],dtype=float)


#init placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#define functions-------------------------------------------------------

#create tf.variables for weights and biases:
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)
  
#layer operations:  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                                           
#init input-------------------------------------------------------------
np.set_printoptions(suppress=True)
np.set_printoptions(precision=15)
np.random.seed(2)

#init: weights/biases---------------------------------------------------
np.random.seed(2)
init2s = np.random.rand(5,5).astype('f') #as float32
init2= init2s.reshape(5,5,1,1)

np.random.seed(3)
init3s = np.random.rand(5,5).astype('f') #as float32
init3= init3s.reshape(5,5,1,1)

np.random.seed(4)
init4 = np.random.rand(4*4,10).astype('f')

#define model parameter-------------------------------------------------
#1.Layer: 1.conv layer                
W_conv1 = tf.Variable(init2)
b_conv1 = bias_variable([1])
#2.Layer: 2.conv layer                  
W_conv2 = tf.Variable(init3)
b_conv2 = bias_variable([1])
#3.Layer: Fc layer
W_fc1 = tf.Variable(init4)
b_fc1 = bias_variable([10])

#list of variables
weightbiaslist = [W_conv1,b_conv1,W_conv2,b_conv2, W_fc1,b_fc1]

# reshape input
x_image = tf.reshape(x, [-1,28,28,1])
#~ x_view =  tf.reshape(x_image, [28,28]) #for view only

########################################################################
#define CNN model
########################################################################

#1.Layer: conv----------------------------------------------------------
h_conv1 = conv2d(x_image, W_conv1) + b_conv1
h_relu1 = tf.nn.relu(h_conv1)
h_pool1 = max_pool_2x2(h_relu1)

#2.Layer: conv----------------------------------------------------------
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_relu2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool_2x2(h_relu2)

#3.Layer: Densely Connected Layer---------------------------------------
h_flat = tf.reshape(h_pool2, [-1, 4*4*1])
y_conv = tf.matmul(h_flat, W_fc1) + b_fc1

#4.Layer: cross_entropy ------------------------------------------------
softmax = tf.nn.softmax(y_conv)
#Cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

########################################################################
#backward pass:
########################################################################

#-----------------------------------------------------------------------
#do manual backprop
#-----------------------------------------------------------------------
# Create an optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate = learnrate)
# Compute the gradients for a list of variables.
grads_and_vars = opt.compute_gradients(cross_entropy, weightbiaslist)
# apply the gradients
apply_grads = opt.apply_gradients(grads_and_vars)
#-----------------------------------------------------------------------

#init testing:
#compare labels and predictions
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#tf.cast: converts float64 to float32
#take the mean of the correct_prediction vector
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#init session:----------------------------------------------------------
sess = tf.InteractiveSession()
#sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())

########################################################################
#train model:
########################################################################

t1_1 = datetime.datetime.now()
print('=============================================')
print("start CNN training" )
print('=============================================')

for epoch in range(1,epochs+1):
	
	for batch in range(1,batches_train+1):
		
		#grab train batch:
		trainbatchx = imagetrainxx[:,batch-1:batch+batsize_train-1] 
		#reshape the input training batch:
		imt5 = np.empty(shape=[1,784],dtype=float)
		
		for pos2 in range(batsize_train):
			imt = trainbatchx[:,pos2]
			imt2 = np.reshape(imt,(28, 28))
			imt3 = np.transpose(imt2)
			imt4 = np.reshape(imt3,(1, 784))
			imt5 = np.append(imt5,imt4, axis=0)
			#~ plt.figure(pos2+10)
			#~ plt.axis("off")
			#~ imgplot11 = plt.imshow(imt3)
		
		#batch: train images and train labels	
		trainbatch = imt5[1:batsize_train+1,:] 
		trainbatchlab = labtrainone[batch-1:batch+batsize_train-1,:] 
		feedtrain = {x:trainbatch , y_: trainbatchlab}
		
		################################################################
		#printout: input, weights and biases
		################################################################
		
		#~ print(trainbatch.shape)
		#~ print ("inputlabel:\n",trainbatchlab)
		#~ print ("input:")
		
		#show image:
		#~ imttx = np.reshape(trainbatch[2,:],(28, 28))
		#~ plt.figure(25)
		#~ plt.axis("off")
		#~ imgp = plt.imshow(imttx)
		
		#~ print ("Weight1:")
		#~ print (sess.run(W_conv1))
		#~ print ("bias1:")
		#~ print (sess.run(b_conv1))
		#~ print ("Weight2:")
		#~ print (sess.run(W_conv2))
		#~ print ("bias2:")
		#~ print (sess.run(b_conv2))
		#~ print ("Weight3FC:")
		#~ print (sess.run(W_fc1))
		#~ print ("bias3fc:")
		#~ print (sess.run(b_fc1))
		################################################################
		
		################################################################
		#printout: forward pass
		################################################################
		
		#1.Layer--------------------------------------------------------
		#~ print ("1.Layer:--------------------------------")
		#~ print (sess.run(h_conv1,feed_dict= {x: trainbatch}))
		#~ conv1y = sess.run(h_conv1,feed_dict= {x: trainbatch})
		#~ conv1 = tf.reshape(conv1y, [24, 24])
		#~ print ("h_conv1:")
		#~ print (sess.run(conv1))
		#~ relu1y = sess.run(h_relu1,feed_dict= {x: trainbatch})
		#~ relu1 = tf.reshape(relu1y, [24, 24])
		#~ print ("h_relu1:")
		#~ print (sess.run(relu1))
		#~ pool1y = sess.run(h_pool1,feed_dict= {x: trainbatch})
		#~ pool1 = tf.reshape(pool1y, [12, 12])
		#~ print ("h_pool1:")
		#~ print (sess.run(pool1))
		#---------------------------------------------------------------
		
		#2.Layer--------------------------------------------------------
		#~ print ("2.Layer:--------------------------------")
		#~ print (sess.run(h_conv1,feed_dict= {x: trainbatch}))
		#~ conv2y = sess.run(h_conv2,feed_dict= {x: trainbatch})
		#~ conv2 = tf.reshape(conv2y, [8, 8])
		#~ print ("h_conv2:")
		#~ print (sess.run(conv2))
		#~ relu2y = sess.run(h_relu2,feed_dict= {x: trainbatch})
		#~ relu2 = tf.reshape(relu2y, [8, 8])
		#~ print ("h_relu2:")
		#~ print (sess.run(relu2))
		#~ pool2y = sess.run(h_pool2,feed_dict= {x: trainbatch})
		#~ pool2 = tf.reshape(pool2y, [4, 4])
		#~ print ("h_pool2:")
		#~ print (sess.run(pool2))
		#---------------------------------------------------------------
		
		#3.Layer--------------------------------------------------------
		#~ print ("3.Layer:--------------------------------")
		#~ print ("fc matmul check:")
		#~ fctest = sess.run(y_conv,feed_dict= {x: trainbatch})
		#~ print (fctest)
		
		#~ print ("softmax check:")
		#~ softmaxtest = sess.run(softmax,feed_dict= {x: trainbatch})
		#~ print (softmaxtest)
		#---------------------------------------------------------------
		
		################################################################
		#history get backprop values
		################################################################
		##  get the gradients (for update)
		grad_values = sess.run([grad for (grad,var) in grads_and_vars], feed_dict=feedtrain)
		gradvar_history = np.append(gradvar_history,grad_values)
		##  get the weights (var)
		var_values = sess.run([var for (grad,var) in grads_and_vars], feed_dict=feedtrain)
		applygrad_history = np.append(applygrad_history,var_values)
		
		################################################################
		#printout: backward pass
		################################################################
		
		#~ crosscheck = sess.run(cross_entropy,feed_dict=feedtrain)
		#~ print("cross entropy check:\n", crosscheck)
		
		#print gradients:-----------------------------------------------
		#~ print ('grad_values weight1: ')
		#~ print (sess.run(tf.reshape(grad_values[0], [5, 5])))
		#~ print ('grad_values bias1 : ')
		#~ print (grad_values[1])
		#~ print ('grad_values weight2: ')
		#~ print (sess.run(tf.reshape(grad_values[2], [5, 5])))
		#~ print ('grad_values bias2 : ')
		#~ print (grad_values[3])
		#~ print ('grad_values weight3fc: ')
		#~ print (sess.run(tf.reshape(grad_values[4], [4*4, 10])))
		#~ print ('grad_values bias3 : ')
		#~ print (grad_values[5])
		#~ print('=============================================')
		#~ print('=============================================')
		
		####################################################################
		
		if batch%100 == 0:
			train_accuracy = accuracy.eval(feed_dict=feedtrain)
			crossloss = sess.run(cross_entropy,feed_dict=feedtrain)
			t2_1 = datetime.datetime.now()
			print('epoch: '+ str(epoch)+'/'+str(epochs)+
			' -- training batch: '+ str(batch)+'/'+str(batches_train)+
			" -- cross entropy loss: " + str(crossloss))
			print('training accuracy: %.2f'% train_accuracy + 
			" -- training time: " + str(t2_1-t1_1))
		
		##weight update: Minibatch - SGD
		train_step = sess.run(apply_grads, feed_dict=feedtrain)
		
		#get cost_history, accuracy history data:
		cost_history = np.append(cost_history,sess.run(cross_entropy,feed_dict=feedtrain))
		acc_history = np.append(acc_history,sess.run(accuracy,feed_dict=feedtrain))
       
#Measure Training Time:-------------------------------------------------       
t3_1 = datetime.datetime.now()
print('---------------------------------------------')
print("overall training time: " + str(t3_1-t1_1)+'\n')


########################################################################
#testing different datasets:
########################################################################
  
#######################Testing: MNIST data##############################

print('=============================================')
print('start CNN testing')
print('=============================================')

for pos in range(0,test_size,batsize_test):
	testbatchx = imagetestxx[:,pos:pos+batsize_test] #grab test batch
	
	#reshape the input test batch:
	imtx5 = np.empty(shape=[1,784],dtype=float)
	
	for pos2 in range(batsize_test):
		imtx = testbatchx[:,pos2]
		imtx2 = np.reshape(imtx,(28, 28))
		imtx3 = np.transpose(imtx2)
		imtx4 = np.reshape(imtx3,(1, 784))
		imtx5 = np.append(imtx5,imtx4, axis=0)
		#~ plt.figure(pos2+30)
		#~ plt.axis("off")
		#~ imgplot11 = plt.imshow(imtx3)
		
	testbatch = imtx5[1:batsize_test+1,:] #batch: test images	
	testbatchlab = labtestone[pos:pos+batsize_test,:] #batch: test labels
	feedtest = {x: testbatch , y_: testbatchlab}
	
	predictions = accuracy.eval(feed_dict=feedtest)
	correctbatch = np.append(correctbatch,predictions)
	if pos%100 == 0:
		print('test batch: '+str(pos+batsize_test)+'/'+str(test_size)+
		' -- test accuracy: %.2f' % np.mean(correctbatch))
print('---------------------------------------------')
print("overall test accuracy %.3f"%np.mean(correctbatch, axis=0)+'\n') 


########################################################################
#plot settings:
########################################################################

#=======================================================================
#plot training
#=======================================================================
#plot loss function-----------------------------------------------------
plt.figure(1, figsize=(8,8))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(cost_history)),cost_history, color='b')
plt.axis([0,batches_train,0,10])
plt.title('Cross Entropy Training Loss')
plt.xlabel('number of batches')
plt.ylabel('loss')
#plot training accuracy function----------------------------------------
plt.subplot(212)
plt.plot(range(len(acc_history)),acc_history, color='g')
plt.axis([0,batches_train,0,1])
plt.title('Training Accuracy')
plt.xlabel('number of batches')
plt.ylabel('accuracy percentage')

#plt.show()
#plt.hold(False)

#=======================================================================
#plot testing
#=======================================================================
#plot loss function-----------------------------------------------------
plt.figure(2, figsize=(8,8))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(cost_history)),cost_history, color='b')
plt.axis([0,batches_train,0,np.max(cost_history)])
plt.title('Cross Entropy Training Loss')
plt.xlabel('number of batches')
plt.ylabel('loss')
#plot test accuracy function----------------------------------------
plt.subplot(212)
plt.plot(np.arange(1,batches_test +1, 1.0),correctbatch, color='m')
#plt.axis([int(1),int(numtestbat),0,1])
plt.yticks(np.arange(0,1.2, 0.2))
plt.xticks(np.arange(1,batches_test +1, 1.0))
plt.title('Test Accuracy')
plt.xlabel('number of batches, batch size: '+ str(batsize_test))
plt.ylabel('accuracy percentage')


#plt.ion()
plt.show()
#plt.hold(False)



