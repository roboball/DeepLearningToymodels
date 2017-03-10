function [ imuppool] = meanpoolup( impool, pstride )
%Upsampling: Mean Pooling

pvec=pstride*pstride;
[m,n] = size(impool);
psize= pstride * m;
imuppool = zeros(psize);
imuppoolint = zeros(m);
for i=1:m
    for j=1:n
        impoolmean = impool(i,j)/pvec;
        imuppoolint(i,j) = impoolmean;
    end
end

i=1;
for k = 1:pstride:psize-1 
    j=1;
    for l = 1:pstride:psize-1
        imuppool(k:k+pstride-1,l:l+pstride-1)=imuppoolint(i,j);
        j=j+1;
    end
    i=i+1;
end

end

