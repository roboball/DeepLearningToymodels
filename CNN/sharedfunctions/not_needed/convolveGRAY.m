function [ im1filgray ] = convolveGRAY( filter,im1gray)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1. 2D convolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im1filgray = conv2(double(im1gray),double(filter),'same' );
%figure(9),imshow(im1filgray)
end

