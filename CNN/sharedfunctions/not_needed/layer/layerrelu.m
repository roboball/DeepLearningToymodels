function [ output_relu] = layerrelu( input )
    %computes the activation layer
    threshold=0;
    output_relu = input;
    output_relu(input < threshold) = 0;

end

