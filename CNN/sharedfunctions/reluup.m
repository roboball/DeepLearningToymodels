function [ uprelu ] = reluup( hiddendelta, forwardpass )
%relu up calculates the delta of the relu activation function
threshold=0;
imuprelu = forwardpass;
imuprelu(forwardpass > threshold) = 1;
imuprelu2 = bsxfun(@times,imuprelu,hiddendelta);
uprelu = imuprelu2;
end

