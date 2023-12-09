load num_cells_3.mat
inputdata_3 = x;
uVectors_3 = u;
load num_cells_5.mat
inputdata_5 = x;
uVectors_5 = u;
load num_cells_10.mat
inputdata_10 = x;
uVectors_10 = x;
load num_cells_20.mat
inputdata_20 = x;
uVectors_20 = u;

nnet_3 = feedforwardnet([64 32]);
nnet_5 = feedforwardnet([64 32]);
nnet_10 = feedforwardnet([64 32]);
nnet_20 = feedforwardnet([64 32]);
inputSize = 1000;
% determine the net to use


% Pick the random 900 then the remaining 100 will be trained

nnet_3 = train(nnet_3, inputdata_3(:, 1:(inputSize*.9)), uVectors_3(:, 1:(inputSize*.9)));
nnet_5 = train(nnet_5, inputdata_5(:, 1:(inputSize*.9)), uVectors_5(:, 1:(inputSize*.9)));
nnet_10 = train(nnet_10, inputdata_10(:, 1:(inputSize*.9)), uVectors_10(:, 1:(inputSize*.9)));
nnet_20 = train(nnet_20, inputdata_20(:, 1:(inputSize*.9)), uVectors_20(:, 1:(inputSize*.9)));
% plot the error over time based on uvector vs output from nn

%save newNet128_64_16.m nnet
maxErr = zeros(4, 1);
avgErr = zeros(4, 1);
outputTest = zeros(3, 4, 1);
actual = zeros(3, 4, 1);
sqErr = zeros(4, 1);
%badGuess = zeros(numcell, 1);
%badinput = zeros(numcell, 1);
%actualsol = zeros(numcell, 1);

% split these for each network and output them
for i=(inputSize*.9+1):inputSize
    % plot it here
    outputTest(:, 1) = nnet_3(inputdata_3(:,i));
   % outputTest(:, 2) = nnet_5(inputdata_5(:,i));
    %outputTest(:, 3) = nnet_10(inputdata_10(:,i));
    %outputTest(:, ) = nnet_20(inputdata_20(:,i));

    actual(:, 1) = uVectors_3(:,i);
   % actual(:,2) = uVectors_5(:,i);
   % actual(:,3) = uVectors_10(:,i);
   % actual(:,4) = uVectors_20(:,i);

    for j=1:1
        sqErr(j) = rms(actual(j) - outputTest(j));
        avgErr(j) = avgErr(j) + sqErr(j);
    end
end

for k=1:1
    display("Avg Error: " + avgErr(k)/(inputSize*.1));
    display("Max Error: " + maxErr(k));
    display(actual(k));
end
maxErr