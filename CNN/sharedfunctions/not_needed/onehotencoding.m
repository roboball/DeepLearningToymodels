function [ onehotoutput ] = onehotencoding( labels, numofclass )
%one hot encoding: convert labels to targets
%number of classes: e.g. 10 digits (1,2,..,9,0)
onehotoutput = single(zeros(numofclass,length(labels)));
for i = 1:length(labels)
    if labels (i) == 0
        labels (i) = 10;
        onehotoutput(labels(i),i) = 1;
    else
        onehotoutput(labels(i),i) = 1;
    end
end

end
 
