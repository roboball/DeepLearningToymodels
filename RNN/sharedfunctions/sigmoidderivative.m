function [ outputderivative ] = sigmoidderivative( output )
    %calculates the derivative of a sigmoid function
    outputderivative = times(output,(1-output));
end

