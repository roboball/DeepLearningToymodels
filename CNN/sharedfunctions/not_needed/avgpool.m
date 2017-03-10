function [ impool1mat ] = avgpool( imcrop, pstride )
%max pooling layer 

pvec=pstride*pstride;
[m,n]=size(imcrop);
impool1mat =zeros(m/pstride);
i=1;
for k = 1:pstride:m-1 
    j=1;
    for l = 1:pstride:n-1
        impool1 =imcrop(k:k+pstride-1,l:l+pstride-1);
        impool1 = reshape(impool1,[pvec,1]);
        impool1mat (i,j)=mean(impool1);
        j=j+1;
    end
    i=i+1;
end

end

