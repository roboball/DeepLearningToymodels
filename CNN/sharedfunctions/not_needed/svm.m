function [ imloss ] = svm( fcvec )
%support vector machine (SVM) calculate the hinge loss function

imloss = max(0, fcvec(1)-fcvec(3)+1)+ max(0, fcvec(2)-fcvec(3)+1)

end

