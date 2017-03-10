function [ imcrop ] = imcrop(crop,imagest10k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%28x28 MNIST gray image crop to 24x24 window
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A=imagest10k(:,1:1);
im = reshape(A,[28,28]);
[m,n]=size(im);
imcrop =im(1+crop:m-crop,1+crop:n-crop);
end

