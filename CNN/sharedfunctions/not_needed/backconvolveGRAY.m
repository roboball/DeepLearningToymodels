function [ imerror, imgrad ] = backconvolveGRAY( imforward, weights,error)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1. Backpropagation: 2D convolution layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%https://cs.nyu.edu/~gwtaylor/publications/nips2010/gwtaylor_nips2010_supp.pdf

%Gradient convolutional layer
%needed for update of weights (dE/dw,(Cross-Correlation)
%flip = rot90(..,2) = 180°Rotation
imgrad = flip(conv2(double(imforward),double(rot90(error,2)),'valid' ));
%figure(10),imshow(imgrad)

%Backpropagation error signal 
%deltas d: dE/dimforward,(Cross-Correlation)
imerror = conv2(double(error),double(rot90(weights,2)),'full' );
%figure(11),imshow(imerror)



end

is my assumption correct that we only flip the gradient Error w.r.t. weigths (dE/dw)?  the gradient for the error signal is not flipped (dE/dInput)?
calculate gradient (for update): gradient = conv2(iminput,rot90(error,2)),'valid' );
calculate error signal: errorgradient = conv2( error, weights,'full' );
so the equations on slide #10 are wrong?:
Backpropagation in Convolutional Neural Network (http://de.slideshare.net/kuwajima/cnnbp)
