function [ impoolup ] = maxpoolup( dEdz,storepool, pstride )
% maxpoolup upsamples pooled gradient
[dim1, dim2, dim3] = size(storepool);
dimup = pstride*dim1;
impoolup = zeros(dimup,dimup,dim3); %creates upsampled template
for w3 = 1:dim3
    k=1;
    for w1 = 1:dim1
        for w2 = 1:dim2
            matrixpart = zeros(pstride*pstride,1);
            matrixpart(storepool(w1,w2)) = dEdz(w1,w2);
            matrixpart2 = reshape(matrixpart, pstride,pstride);
            impoolup(pstride*w1-(pstride-1):pstride*w1,k:k+pstride-1,w3) = matrixpart2;
            k=k+pstride;
        end
        k=1;
    end
end
    
end

