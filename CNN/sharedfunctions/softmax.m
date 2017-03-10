function [ scores, dataloss, delta ] = softmax( ysoft, target )

%softmax classifier 
scores = bsxfun(@rdivide, exp(ysoft), sum(exp(ysoft)));
% imexp   = exp(ysoft); %exp function
% sumexp  = sum(imexp); %sum of exp
% y = (1./sumexp) .* imexp; %normalization, gives softmax prob

%cross-entropy loss function
%(multinomial logistic classification)

%calculates the data loss:
%numsamples = length(ysoft);
numsamples = 1;
%dataloss = - (1/numsamples)* sum(times(target(1,:),log(scores))); %für zeilenvektor target
dataloss = - (1/numsamples)* sum(times(target(:,1),log(scores))); %for column vector target

% %calculate the regularization loss (L2):
% regloss = 0.5 * lambda * sum(sum(weights.^2));
% 
% %get totalloss:
% totalloss = dataloss + regloss;

%backprob:--------------------------------------------------
%https://www.ics.uci.edu/~pjsadows/notes.pdf
delta = scores - target;
delta = delta';


end
 
