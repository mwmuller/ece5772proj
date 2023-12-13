numcell = 4;
inputSize = 10000;
sampling_time = 10;
uMax = 0.3;
beta = sampling_time/(3600*4.1);
balanceArr = zeros(inputSize, 1); % 1 = balance value ; 2 = maxDiff; 3 = diff
maxDiffArr = zeros(inputSize, 1);
maxTimeArr = zeros(inputSize, 1);
uVectors = zeros(inputSize, numcell);
% creating test data
inputdata = randi([1, 100], inputSize, numcell) ./100;
inputdata(1, :) = [0.0300 ,   0.9500 ,   0.9000  ,  0.0800];
%inputdata(2, :) = [1.0, 0.4, 0.6];
% find the balanced value

for i=1:inputSize
    % 
    balanceArr(i) = sum(inputdata(i, :))./numcell;
    % calc the temp maxdiff
    maxDiffArr(i) = max(abs(balanceArr(i) - inputdata(i, :)));
    
    % if the max is negative, check for other negative maxes otherwise
    % check fo

    balDiff = balanceArr(i) - inputdata(i, :);

    getmaxes = find(max(balDiff) == balDiff);
    getmins = find(min(balDiff) == balDiff);
    minSize = size(getmins,2);
    maxSize = size(getmaxes, 2);
    if minSize == maxSize
        if abs(balDiff(getmins)) ~= abs(balDiff(getmaxes))
            % if they are not the same, get the max and take the size
            if abs(balDiff(getmins)) > abs(balDiff(getmaxes))
                count = minSize;
            else
                count = maxSize;
            end
        end
    else
        if abs(balDiff(getmins)) > abs(balDiff(getmaxes))
            count = minSize;
        else
            count = maxSize;
        end
    end
    if mod(count , 2) == 0
        if sign(balDiff(getmaxes(1))) ~= sign(balDiff(getmaxes(2)))
            count = 1;
        end
    end
    %calc the max time for balance
    maxTimeArr(i) = maxDiffArr(i)/(beta*(uMax/(count)));

    uVectors(i, :) = (balanceArr(i) - inputdata(i, :))/((maxTimeArr(i)*beta));
end