function [ imrelu ] = relu( imcrop )
%relu calculates the activation function for each neuron
threshold=0;
imrelu = imcrop;
imrelu(imcrop < threshold) = 0;

end

