%==========================================================================
%Title:    Binary Recurrent Neural Network (RNN)
%Purpose:  basic RNN that predicts the addition of 2 binary digits
%
%Equation: a +  b =  d ?? compare to true label  c => Error = ??
%          8 + 16 = 25 ?? compare to true label 24 => Error = 1
%
%          a and b are inputs and are added up to d. Then both are 
%          are compared to the true label c. The error gets then
%          backpropagated via the Backpropagation Through Time
%          (BPTT) algorithm.
%          
%Inputs:   a, 8 bit binary vector, e.g.   [0,0,0,0,1,0,0,0] =    8
%          b, 8 bit binary vector, e.g. + [0,0,0,1,0,0,0,0] = + 16 
%                                  ---------------------------------
%Output:   d, 8 bit binary vector, e.g.   [0,0,0,1,1,0,0,0] =   25
%Label:    c, 8 bit binary vector, e.g. - [0,0,0,1,1,0,0,1] = - 24 
%                                  ---------------------------------
%                                                     Error =    1
%
%Version:  2/2017 Roboball     
%Links:    https://github.com/roboball/DeepLearningToymodels
%==========================================================================

clear all
close all
clc

%add filepath and sub directories
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
addpath(genpath('../RNN'));

%optional: fix random number generator
%rng(2); 

%training dataset generation
binary_dim = 8;

%create binary numbers:
largest_number = 2 ^ binary_dim;
binfold = linspace(0,255,256)';
binary = fliplr(de2bi(binfold,8));

%input variables
alpha = 0.1; 
input_dim = 2;
hidden_dim = 16;
output_dim = 1;
%for 500,000 epochs 
%get training acc.: ca. 99.4 %
%training time: 10 min (icore7)
epochs = 20000; 

%init error, training accuracy
toterror = [];
trainacc =[];
acc_OK = 0; acc_False = 0; rel_acc_OK = 0; 

% initialize neural network weights
synapse_0 = 2*rand(input_dim,hidden_dim)-1;  %U(2x16)
synapse_1 = 2*rand(hidden_dim,output_dim)-1; %V(16x1)
synapse_h = 2*rand(hidden_dim,hidden_dim)-1; %W(16x16)

synapse_0_update = synapse_0 * 0; %init U_update with zeros
synapse_1_update = synapse_1 * 0; %init V_update with zeros
synapse_h_update = synapse_h * 0; %init W_update with zeros

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start training loop (use at least iterations: 10000)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
for numepochs = 1:epochs
    
    %generate a addition problem (a + b = c)
    %be careful to generate rand numbers a,b < 128
    
    a_int = randi([0 largest_number/4 ]); % int version, use 2
    a = fliplr(de2bi(a_int,8)); % binary encoding
  
    b_int = randi([0 largest_number/2 ]); % int version
    b = fliplr(de2bi(b_int,8)); % binary encoding

    %true answer (label)
    c_int = a_int + b_int;
    c = fliplr(de2bi(c_int,8)); % binary encoding
      
    %init different paramters:
    d = c * 0; % init prediction d with zeros
    overallError = 0; %init overallError
    
    layer_2_deltas = []; %init output deltas
    layer_1_values = []; %init hidden states
    
    %init hidden state(t=0) with zeros
    layer_1_values = [layer_1_values, zeros(1,16)]; 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % start forward pass
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for position = 0:binary_dim-1
        
        %------------------------------------------------------------------
        %X = [a,b] 2 bits, feed into from right to left (LSB to MSB)
        %y = target [c] 1 bit, feed into from right to left (LSB to MSB)
        %layer_0: input layer
        %layer_1: hidden layer
        %layer_2: output layer
        %------------------------------------------------------------------
        
        % generate input and output
        % X and y start with LSB to MSB
        X = [a(binary_dim - position), b(binary_dim - position)];
        y = c(binary_dim - position)';

        %hidden layer (input ~+ prev_hidden)
        l11 = X * synapse_0; %input * U
        %hidden state(t-1) * W  
        l12 = layer_1_values(position*16+1:position*16+16) * synapse_h;   
        layer_1_int = l11 + l12; %hidden state(t) before activation
        layer_1 = sigmoid( layer_1_int );  %hidden state(t)
        
        %output layer (new binary representation)
        layer_2_int = layer_1 * synapse_1;
        layer_2 = sigmoid( layer_2_int ); %output(t)


        %loss at each node: (y=target(t), layer_2=output(t))
        layer_2_error = y - layer_2;
        
        
        %output delta(t) at each node
        outputderivative = sigmoidderivative( layer_2 );
        layer_2_deltas_int = layer_2_error *  outputderivative;
        layer_2_deltas = [layer_2_deltas, layer_2_deltas_int];
        
        %total error (sum of all errors)
        overallError = overallError + abs(layer_2_error);
        
        %decode estimate so we can print it out
        d(binary_dim - position) = round(layer_2);
        
        %store hidden layer for the next timestep
        layer_1_values = [layer_1_values, layer_1];
        
    %init hidden layer delta (t+1) with zeros
    future_layer_1_delta = zeros(1,hidden_dim);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % start backward pass: BPTT
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for position = 0:binary_dim-1
        
        %input X(t)
        X = [a(position+1), b(position+1)];
        %hidden state(t)
        layer_1 = layer_1_values((binary_dim-position)*16+1:(binary_dim-position)*16+16);
        %hidden state(t-1)
        prev_layer_1 = layer_1_values((binary_dim-1-position)*16+1:(binary_dim-1-position)*16+16);
       
        %output delta(t)
        layer_2_delta = layer_2_deltas(binary_dim-position);
        
        %hidden delta(t)
        outputderivative1 = sigmoidderivative( layer_1 );
        
        %outputderivative1 = ones(1,16); %noch falshc!!!
        layer_1_delta = times((future_layer_1_delta * synapse_h' +...
                        layer_2_delta * synapse_1'), outputderivative1);
       
        %gradients for update
        
        %dE/dV = sum(hiddenstate(t)^T * outputdelta(t))
        synapse_1_update = synapse_1_update + layer_1' * layer_2_delta; 
        %dE/dW = sum(hiddenstate(t-1)^T *hiddendelta(t))
        synapse_h_update = synapse_h_update + prev_layer_1' * layer_1_delta; 
        %dE/dU = summ(inputX(t)^T * hiddendelta(t))
        synapse_0_update = synapse_0_update + X' * layer_1_delta;            
        
        %hiddendelta(t+1)
        future_layer_1_delta = layer_1_delta;
    end
    
    % weight update: with summed up gradients, learning rate = alpha
    synapse_0 = synapse_0 + synapse_0_update * alpha;
    synapse_1 = synapse_1 + synapse_1_update * alpha;
    synapse_h = synapse_h + synapse_h_update * alpha;

    %re-init updates to 0 
    synapse_0_update = synapse_0_update * 0;
    synapse_1_update = synapse_1_update * 0;
    synapse_h_update = synapse_h_update * 0;
    
    %update totalerror, training accuracy
    toterror = [toterror,overallError];
    
    %check training accuracy: output d == label c ?
    comp_dc = bsxfun(@eq,d,c);
    sumcomp_dc = sum(comp_dc);
    
    if  sumcomp_dc == 8     
        acc_OK = acc_OK + 1;
        rel_acc_OK = acc_OK / numepochs;
        trainacc = [trainacc, rel_acc_OK];
        %disp('OK')
    else
        acc_False = acc_False +1;
        trainacc = [trainacc, rel_acc_OK];
        %disp('False')
    end
     
    %print out progress after each 1000 epochs
    modulo_ep = mod(numepochs,1000);
    
    if modulo_ep == 0 
        disp('     ')
        disp('----------------------------------------') 
        disp(['Training epochs: ' num2str(numepochs)]) 
        disp('----------------------------------------')
        disp(['Training Time: ' num2str(toc/60,2) ' min'])
        disp(['Error: ' num2str(overallError)])
        disp(['Training Accuracy: ' num2str(rel_acc_OK)])
        disp(['Pred: ' num2str(d)])
        disp(['True: ' num2str(c)])
        out = 0;
        out1=0;
        
        %convert result: binary to decimal
        for index = 0:length(d)-1
              x = d(length(d) - index);   
              out = out + x * (2 ^ index);
        end
        %display result in decimal
        disp([num2str(a_int) ' + ' num2str(b_int) ' = ' num2str(out)])
         
     end

end
disp('   ')
disp(['Total Training Time: ' num2str(toc,2),' sec'])
%disp(['Total Training Time: ' num2str(toc/60,2),' min'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot settings: total error, training accuracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot total error:
hFig = figure(1);
set(hFig, 'Position', [600 200 800 600]) % [x y width height]
maxiloss = max(toterror);
subplot(2,1,1);
plot(toterror,'b','LineWidth',1);
axis([0 numepochs+1 0 ceil(maxiloss)+2]);
title(['\fontsize{14} Training: RNN after ',num2str(numepochs),' epochs ', ...
       '\newline \fontsize{10}  \newline \fontsize{12}                  Training Error ['...
         num2str(overallError,3) ']']);
%title(['\fontsize{12} Total Training Error: RNN after ',num2str(numepochs),' epochs ', ...
%        '\newline \fontsize{10}                  RNN for addition of binary digits']);
%xlabel('epochs')
ylabel('training error')

%plot training accuracy:
%figure(2)
subplot(2,1,2);
plot(trainacc,'g','LineWidth',2);
axis([0 numepochs+1 0 1]);
title(['\fontsize{10}  ', ...
       '\newline \fontsize{12} Training Accuracy [' num2str(rel_acc_OK,3) ']']);
xlabel('epochs')
ylabel('training accuracy in %')


%==========================================================================


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%number of testsamples
testsamples = 10000;
%init test parameter
toterror2 = []; testacc=[]; rel_acc_OK2=0; acc_OK2 = 0;acc_False2=0;

tic
for numepochs = 1:testsamples
    
    %generate testsamples a and b  
    a_int = randi([0 largest_number/4 ]); % int version, use 2
    a = fliplr(de2bi(a_int,8)); % binary encoding
  
    b_int = randi([0 largest_number/2 ]); % int version
    b = fliplr(de2bi(b_int,8)); % binary encoding

    %true answer (label)
    c_int = a_int + b_int;
    c = fliplr(de2bi(c_int,8)); % binary encoding
      
    %init different paramters:
    d = c * 0; % init prediction d with zeros
    overallError2 = 0; %init overallError
    
    layer_2_deltas = []; %init output deltas
    layer_1_values = []; %init hidden states
    
    %init hidden state(t=0) with zeros
    layer_1_values = [layer_1_values, zeros(1,16)]; 
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % start forward pass
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for position = 0:binary_dim-1
        
        %------------------------------------------------------------------
        %X = [a,b] 2 bits, feed into from right to left (LSB to MSB)
        %y = target [c] 1 bit, feed into from right to left (LSB to MSB)
        %layer_0: input layer
        %layer_1: hidden layer
        %layer_2: output layer
        %------------------------------------------------------------------
        
        % generate input and output
        % X and y start with LSB to MSB
        X = [a(binary_dim - position), b(binary_dim - position)];
        y = c(binary_dim - position)';

        %hidden layer (input ~+ prev_hidden)
        l11 = X * synapse_0; %input * U
        %hidden state(t-1) * W  
        l12 = layer_1_values(position*16+1:position*16+16) * synapse_h;   
        layer_1_int = l11 + l12; %hidden state(t) before activation
        layer_1 = sigmoid( layer_1_int );  %hidden state(t)
        
        %output layer (new binary representation)
        layer_2_int = layer_1 * synapse_1;
        layer_2 = sigmoid( layer_2_int ); %output(t)

        %loss at each node: (y=target(t), layer_2=output(t))
        layer_2_error = y - layer_2;
        
        %total error (sum of all errors)
        overallError2 = overallError2 + abs(layer_2_error);
        
        %prediction: d
        d(binary_dim - position) = round(layer_2);
        
        %store hidden layer for the next timestep
        layer_1_values = [layer_1_values, layer_1];
        
    end
    
    %update totalerror, training accuracy
    toterror2 = [toterror2,overallError2];
    
    %check training accuracy: output d == label c ?
    comp_dc = bsxfun(@eq,d,c);
    sumcomp_dc = sum(comp_dc);
    
    if  sumcomp_dc == 8     
        acc_OK2 = acc_OK2 + 1;
        rel_acc_OK2 = acc_OK2 / numepochs;
        testacc = [testacc, rel_acc_OK2];
        %disp('OK')
    else
        acc_False2 = acc_False2 +1;
        testacc = [testacc, rel_acc_OK2];
        %disp('False')
    end
     
    %print out progress after each 1000 epochs
    modulo_ep = mod(numepochs,1000);
    
    if modulo_ep == 0 
        disp('     ')
        disp('----------------------------------------') 
        disp(['Test samples: ' num2str(numepochs)]) 
        disp('----------------------------------------')
        disp(['Test Time: ' num2str(toc/60,2) ' min'])
        disp(['Error: ' num2str(overallError2)])
        disp(['Training Accuracy: ' num2str(rel_acc_OK2)])
        disp(['Pred: ' num2str(d)])
        disp(['True: ' num2str(c)])
        out = 0;
        out1=0;
        
        %convert result: binary to decimal
        for index = 0:length(d)-1
              x = d(length(d) - index);   
              out = out + x * (2 ^ index);
        end
        %display result in decimal
        disp([num2str(a_int) ' + ' num2str(b_int) ' = ' num2str(out)])
         
     end

end
disp('   ')
disp(['Total Test Time: ' num2str(toc,2),' sec'])
%disp(['Total Training Time: ' num2str(toc/60,2),' min'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot settings: total test error, test accuracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot total error:
hFig2 = figure(2);
set(hFig2, 'Position', [600 200 800 600]) % [x y width height]
maxiloss = max(toterror);
subplot(2,1,1);
plot(toterror2,'b','LineWidth',1);
axis([0 numepochs+1 0 ceil(maxiloss)+2]);
title(['\fontsize{14} Testing: RNN after ',num2str(numepochs),' test samples ', ...
       '\newline \fontsize{10}  \newline \fontsize{12}                          Test Error ['...
         num2str(overallError2,3) ']']);
%title(['\fontsize{12} Total Training Error: RNN after ',num2str(numepochs),' epochs ', ...
%        '\newline \fontsize{10}                  RNN for addition of binary digits']);
%xlabel('epochs')
ylabel('test error')

%plot training accuracy:
%figure(2)
subplot(2,1,2);
plot(testacc,'g','LineWidth',2);
axis([0 numepochs+1 0 1]);
title(['\fontsize{10}  ', ...
       '\newline \fontsize{12} Test Accuracy [' num2str(rel_acc_OK2,3) ']']);
xlabel('test samples')
ylabel('test accuracy in %')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% optional: export model weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%export weights to mat file: 
%save(['bin_RNNweights_',num2str(epochs),'_',num2str(rel_acc_OK,2),'.mat'],'synapse_0','synapse_1','synapse_h')





