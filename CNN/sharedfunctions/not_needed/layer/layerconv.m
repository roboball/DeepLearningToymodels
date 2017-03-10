function [ outputvolume ] = layerconv( inputvolume, filter )
%computes convolutional layer
[w,h,m,n] = size(filter);
for i = 1:m
    for g = 1:n
    outputvolumex = conv2(inputvolume(:,:,:,i),rot90(filter(:,:,i,g),2),'valid');
    outputvolumey(:,:,i,g) = outputvolumex;
    end
end

if m == 1
    outputvolume = outputvolumey;
else
    outputvolume = sum(outputvolumey,3);
end

end

