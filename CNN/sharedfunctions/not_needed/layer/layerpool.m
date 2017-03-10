function [ outputpool, storepool ] = layerpool( input, stride )
%computes the pooling layer
%subfunction: maxpool
[w,h,m,n] = size(input);
for i = 1:n
   [ outputpoolx , storepoolx ] =  maxpool(input(:,:,:,i), stride);

   outputpool(:,:,1,i) = single(outputpoolx);
   storepool(:,:,1,i) = single(storepoolx) ;
end

end

