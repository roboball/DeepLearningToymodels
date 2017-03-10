function [ neuronout ] = sigmoid( n )
%Sigmoid calculates the activation function for each neuron
neuronout = 1 ./( 1 + exp(-n));
end

