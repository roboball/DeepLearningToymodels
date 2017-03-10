function [ outputfilter ] = layergradient( inputvolume, delta )
%computes convolutional layer
[w,h,m,n] = size(delta);
[w2,h2,m2,n2] = size(inputvolume);
for i =1:n
    for g = 1:n2
    outputfilter(:,:,g,i) = conv2(inputvolume(:,:,:,g),rot90(delta(:,:,:,i),2),'valid');
    end
end

end

