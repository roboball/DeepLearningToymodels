function [ output ] = sigmoid( n )
    %calculates the sigmoid function
    output = 1 ./( 1 + exp(-n));
end

