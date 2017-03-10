%==========================================================================
%Purpose: Classification of Digits with a CNN
%Model:   3 Layer Convolutional Neural Network
%         Trained with Minibatch Gradient Descent
%Inputs:  MNIST Database (digits)
%         60,000 training set
%         10,000 test set 
%Output:  10 Classes (digits from 1 to 10)
%Version: 10/2016 (roboball MattK.)
%Link: https://github.com/roboball/DeepLearning/blob/master/README.md
%==========================================================================

clear all
close all
clc

%add filepath and sub directories
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
addpath(genpath('../CNN'));

%********************************************************
%inputs from mat file: 
%********************************************************
load('data/MNIST.mat')

%********************************************************
%init parameter:
%********************************************************
epochs = 1; %number of iterations
eta = 0.01; %learning rate 
trainsamples = 60000; %60000 training samples
batsize_train = 50; %training batch size 
testsamples = 10000; %10000 test samples
batsize_test = 2000; %test batch size 
prostep = 100; %stepsize for proof of progress
numclass = 10; %number of classes

%number of batches for training and testing:
batches_train = trainsamples / batsize_train; 
batches_test = testsamples / batsize_test; 

%dummy vector: gradient for nnloss
gradloss = ones(1,1,10,1, 'double'); 
%init loss history
totalloss =[];
%init accuracy history
trainacc = [0];
testacc =[];

%********************************************************
%create target vectors: one hot encoding
%********************************************************

%training targets
tar_train = zeros(numclass,length(labeltrain));
for label = 1:length(labeltrain)
        tar_train(labeltrain (label)+1,label) = 1;
end

%training targets
tar_test = zeros(numclass,length(labeltest));
for label = 1:length(labeltest)
        tar_test(labeltest (label)+1,label) = 1;
end

%********************************************************
%init weights:
%********************************************************

%init weights
rng(2); format long
w1 = rand(5,5)';
rng(3); format long
w2 = rand(5,5)';
rng(4); format long
w3 = rand(10,4*4)';

%init biases
b1=0;
b2=0;
b3=zeros(1,10);

%==========================================================================
%start training:
%==========================================================================
disp('================================')
disp('start CNN training')
disp('================================')
tic
for epoch = 1:epochs

for batch = 1:batches_train
  
%get minibatch:
minibatch = imagetrain(:,batch:batch+batsize_train-1);  %grab data
minibatchlabel = tar_train(:,batch:batch+batsize_train-1); %grab labels

for sample = 1:batsize_train
      
%**************************************************************************
% FORWARD PASS
%**************************************************************************

%input from minibatch:
x1 = double(reshape(minibatch(:,sample),[28,28]));
target = minibatchlabel(:,sample);

%1.Layer
l1_conv  = conv2(x1,rot90(w1,2),'valid' ) + b1; 
l1_relu  = relu( l1_conv );
[ l1_pool , l1_storepool ] = maxpool(l1_relu, 2);

%2.Layer
l2_conv  = conv2(l1_pool,rot90(w2,2),'valid') + b2; 
l2_relu  = relu( l2_conv );
[ l2_pool , l2_storepool ] = maxpool(l2_relu, 2);

%3.Layer: FC1
l3_reshape = reshape(l2_pool',1,4*4);
l3_fclayer = w3' * l3_reshape' + b3';

% Output Layer: softmax probs, cross-entropy loss and output deltas
[ softmaxprobs, ce_loss, deltaout ] = softmax( l3_fclayer, target );

%**************************************************************************
% BACKWARD PASS
%**************************************************************************

%********************************************************
%hidden deltas:
%********************************************************
%3.Layer:
l3_backfclayer = deltaout * w3'; 
l3_backreshape = reshape(l3_backfclayer ,4,4)';
%2.Layer:
l2_backpool = maxpoolup( l3_backreshape, l2_storepool , 2 );
l2_backrelu = reluup( l2_backpool, l2_relu );
l2_backconv = conv2(l2_backrelu,w2,'full'); 
%1.Layer:
l1_backpool = maxpoolup( l2_backconv,l1_storepool, 2 );
l1_backrelu = reluup( l1_backpool, l1_relu );
l1_backconv = conv2(l1_backrelu,w1,'full');

%********************************************************
%gradients for update:
%********************************************************
%weight gradients:
grad_weight3 = l3_reshape' * deltaout;
grad_weight2 = conv2(l1_pool,rot90(l2_backrelu,2),'valid');
grad_weight1 = conv2(x1,rot90(l1_backrelu,2),'valid'); 
%bias gradients:
grad_bias3 = deltaout;
grad_bias2 = sum(sum(1 * l2_backrelu)); 
grad_bias1 = sum(sum(1 * l1_backrelu)); 

%********************************************************
%store: gradients and loss
%********************************************************
%weight gradients:
gradw1(:,:,:,sample) = grad_weight1;
gradw2(:,:,:,sample) = grad_weight2;
gradw3(:,:,:,sample) = grad_weight3;
%bias gradients:
gradb1(:,:,:,sample) = grad_bias1;
gradb2(:,:,:,sample) = grad_bias2;
gradb3(:,:,:,sample) = grad_bias3;

%loss
batchloss(sample) = ce_loss;

end

%********************************************************
%gradient update: Minibatch Gradient Descent:
%********************************************************

%average the gradients:
avg_gradw1 = sum(gradw1,4)./batsize_train;
avg_gradw2 = sum(gradw2,4)./batsize_train;
avg_gradw3 = sum(gradw3,4)./batsize_train;
avg_gradb1 = sum(gradb1,4)./batsize_train;
avg_gradb2 = sum(gradb2,4)./batsize_train;
avg_gradb3 = sum(gradb3,4)./batsize_train;

%weight updates:
w1 = w1 - eta * avg_gradw1; 
w2 = w2 - eta * avg_gradw2;
w3 = w3 - eta * avg_gradw3; 
b1 = b1 - eta * avg_gradb1;
b2 = b2 - eta * avg_gradb2;
b3 = b3 - eta * avg_gradb3';

%********************************************************
%average batchloss:
%********************************************************
loss = sum(batchloss)/batsize_train;
epochloss(batch) = loss;

%check training accuracy after every 100 batches
if mod(batch,prostep) == 0
%********************************************************
%training accuracy:
%********************************************************
correct=0;
for sample = 1:batsize_train
      
%**************************************************************************
% FORWARD PASS
%**************************************************************************

%input from minibatch:
x1 = double(reshape(minibatch(:,sample),[28,28]));
target = minibatchlabel(:,sample);

%1.Layer
l1_conv  = conv2(x1,rot90(w1,2),'valid' ) + b1; 
l1_relu  = relu( l1_conv );
[ l1_pool , l1_storepool ] = maxpool(l1_relu, 2);

%2.Layer
l2_conv  = conv2(l1_pool,rot90(w2,2),'valid') + b2; 
l2_relu  = relu( l2_conv );
[ l2_pool , l2_storepool ] = maxpool(l2_relu, 2);

%3.Layer: FC1
l3_reshape = reshape(l2_pool',1,4*4);
l3_fclayer = w3' * l3_reshape' + b3';

% Output Layer: softmax probs, cross-entropy loss and output deltas
[ softmaxprobs1, ce_loss1, deltaout1 ] = softmax( l3_fclayer, target );

%compare softmax prob to target
[maxi_softmax, idx] = max(softmaxprobs1); %get index softmax
idxtarget = find(target == 1); %get index target

if idx == idxtarget
    correct = correct + 1;
end

end

%store: training accuracy
trainacc = [trainacc,correct/batsize_train];

%********************************************************
%show CNN training progress:
%********************************************************    
disp(['epoch ' num2str(epoch) '/' num2str(epochs) ' -- training batch '...
        num2str(batch) '/' num2str(batches_train)...
        ' -- cross entropy loss ' num2str(loss)]); 
disp(['training accuracy: ',num2str(correct/batsize_train)...
    ' -- time: ' num2str(toc/60)])
end

end
totalloss = [totalloss,epochloss];
end
traintime = toc/60;
disp('---------------------------------')
disp(['total training time: ',num2str(traintime),char(10) ])

%==========================================================================
%start testing:
%==========================================================================
disp('================================')
disp('start CNN testing')
disp('================================')

for batch2 = 1:batches_test
  
%get minibatch:
minibatch = imagetest(:,batch2:batch2+batsize_test-1);  %grab batch
minibatchlabel = tar_test(:,batch2:batch2+batsize_test-1); %grab labels

correct=0; %init acc
for sample = 1:batsize_test
      
%**************************************************************************
% FORWARD PASS
%**************************************************************************

%input from minibatch:
x1 = double(reshape(minibatch(:,sample),[28,28]));
target = minibatchlabel(:,sample);

%1.Layer
l1_conv  = conv2(x1,rot90(w1,2),'valid' ) + b1; 
l1_relu  = relu( l1_conv );
[ l1_pool , l1_storepool ] = maxpool(l1_relu, 2);

%2.Layer
l2_conv  = conv2(l1_pool,rot90(w2,2),'valid') + b2; 
l2_relu  = relu( l2_conv );
[ l2_pool , l2_storepool ] = maxpool(l2_relu, 2);

%3.Layer: FC1
l3_reshape = reshape(l2_pool',1,4*4);
l3_fclayer = w3' * l3_reshape' + b3';

% Output Layer: softmax probs, cross-entropy loss and output deltas
[ softmaxprobs2, ce_loss2, deltaout2 ] = softmax( l3_fclayer, target );

%compare softmax prob to target
[maxi_softmax2, idx2] = max(softmaxprobs2); %get index softmax
idxtarget2 = find(target == 1); %get index target

if idx2 == idxtarget2
   correct = correct + 1;
end

end

%store: test accuracy
testacc = [testacc,correct/batsize_test];

%********************************************************
%show CNN testing progress:
%********************************************************    
disp(['test batch ' num2str(batch2) '/' num2str(batches_test) ...
      ' -- test accuracy: ',num2str(correct/batsize_test)]) 
end

disp('---------------------------------')
totaltestacc = sum(testacc)/batches_test;
disp(['total test accuracy: ',num2str(totaltestacc),char(10)])

%**************************************************************************
%plot settings: 
%**************************************************************************

%plot: training error
figure(100)
maxiloss = max(totalloss);
plot(totalloss,'b','LineWidth',2);
%axis([1 batch*epochs+0.001 0 ceil(maxiloss)+1])
axis([1 batch*epochs+0.001 0 10])
title(['Training Error: CNN after ',num2str(epochs),' epochs',...
       ' , batch-size: ',num2str(batsize_train)])
xlabel('batches')
ylabel('total error')

if trainsamples ~= 1

%plot: training accuracy
figure(101)
xax = 0:prostep:batch*epochs;
xai = 0:1:batch*epochs;
vq1 = interp1(xax,trainacc, xai, 'pchip');
%plot(xax,trainacc,'g',xai,vq1,'-g','LineWidth',2 );
plot(xai,vq1,'-g','LineWidth',2 );
axis([0 batch*epochs+0.001 0 1])
title('Training Accuracy ')
xlabel('batches')
ylabel('accuracy in %')

end

%plot: test accuracy
figure(102)
xax2 = 1:batch2;
xai2 = 1:1:batch2;
vq2 = interp1(xax2,testacc, xai2, 'pchip');
%plot(xax2,testacc,'m',xax2,vq2,'-m','LineWidth',2);
plot(xax2,vq2,'-m','LineWidth',2);
axis([1 batch2+0.001 0 1])
title('Test Accuracy ')
xlabel('batches')
ylabel('accuracy in %')

















