function [ out ] = selu( x )
% selu activation function
         
         % init globals selu activation function
         ALPHA = 1.6732632423543772848170429916717;
         LAMBDA = 1.0507009873554804934193349852946;
         
         % calculate values
         negative = LAMBDA * ((ALPHA * exp(x)) - ALPHA);
         positive = LAMBDA * x;
         negative (x > 0.0) = 0;
         positive (x <= 0.0) = 0;
         % result
         out = positive + negative;       
end
