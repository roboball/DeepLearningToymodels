function [ impoolmat, location ] = maxpool( imcrop, pstride )
%max pooling layer (stores max values and location)

pvec = pstride*pstride;
[m,n] = size(imcrop);
impoolmat = zeros(m/pstride);
location = zeros(m/pstride);
i=1;
for k = 1:pstride:m-1 
    j=1;
    for l = 1:pstride:n-1
        impool =imcrop(k:k+pstride-1,l:l+pstride-1);
        impool = reshape(impool,[pvec,1]);
        [impoolmat(i,j),location(i,j)]= max(impool);
        j=j+1;
    end
    i=i+1;
end

end

