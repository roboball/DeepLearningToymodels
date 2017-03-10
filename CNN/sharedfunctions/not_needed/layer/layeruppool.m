function [ impoolup ] = layeruppool( dEdz,storepool, stride )
% maxpoolup upsamples pooled gradient
[dim1, dim2, dim3, dim4] = size(storepool);
dimup = stride*dim1;
impoolup = zeros(dimup,dimup,1,dim4); %creates upsampled template
for w3 = 1:dim4
    k=1;
    for w1 = 1:dim1
        for w2 = 1:dim2
            matrixpart = zeros(stride*stride,1);
            matrixpart(storepool(w1,w2,1,w3)) = dEdz(w1,w2,1,w3);
            matrixpart2 = reshape(matrixpart, stride,stride);
            impoolup(stride*w1-(stride-1):stride*w1,k:k+stride-1,1,w3) = matrixpart2;
            k=k+stride;
        end
        k=1;
    end
end
    impoolup = single(impoolup);
end

