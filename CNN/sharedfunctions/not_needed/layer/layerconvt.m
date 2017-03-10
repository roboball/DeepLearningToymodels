function [ outputvolume ] = layerconvt( inputvolume, filter )
%computes convolutional layer
[w,h,m,n] = size(filter);
for i = 1:m
    for g = 1:n
    outputvolumex  = conv2(inputvolume(:,:,:,g),filter(:,:,i,g),'full');
    outputvolumey(:,:,g,i) = outputvolumex;
    end
end
outputvolume = sum(outputvolumey,3);

end

