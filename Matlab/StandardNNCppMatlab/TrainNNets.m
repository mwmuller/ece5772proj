load num_cells_3.mat
inputdata_3 = x;
uVectors_3 = u;
load num_cells_5.mat
inputdata_5 = x;
uVectors_5 = u;
load num_cells_10.mat
inputdata_10 = x;
uVectors_10 = u;
load num_cells_20.mat
inputdata_20 = x;
uVectors_20 = u;

nnet_3_shallow = feedforwardnet(128);
nnet_3_shallow.layers{1}.transferFcn = 'poslin';
nnet_5_shallow = feedforwardnet(128);
nnet_5_shallow.layers{1}.transferFcn = 'poslin';
nnet_10_shallow = feedforwardnet([64 32]);
nnet_10_shallow.layers{1}.transferFcn = 'poslin';
nnet_10_shallow.layers{2}.transferFcn = 'poslin';
nnet_20_shallow = feedforwardnet([64, 32]);
nnet_20_shallow.layers{1}.transferFcn = 'poslin';
nnet_20_shallow.layers{2}.transferFcn = 'poslin';
inputSize = 1000;
% determine the net to use
randidx = randperm(inputSize*.9);

% Pick the random 900 then the remaining 100 will be trained

nnet_3_shallow = train(nnet_3_shallow, inputdata_3(:, randidx), uVectors_3(:, randidx));
nnet_5_shallow = train(nnet_5_shallow, inputdata_5(:, randidx), uVectors_5(:, randidx));
nnet_10_shallow = train(nnet_10_shallow, inputdata_10(:, randidx), uVectors_10(:, randidx));
nnet_20_shallow = train(nnet_20_shallow, inputdata_20(:, randidx), uVectors_20(:, randidx));

%save nnet_3_shallow_v3.mat nnet_3_shallow
%save nnet_5_shallow_v3.mat nnet_5_shallow
%save nnet_10_shallow_v3.mat nnet_10_shallow
%save nnet_20_shallow_v3.mat nnet_20_shallow
%load nnet_3_shallow.mat nnet_3_shallow
%load nnet_5_shallow.mat nnet_5_shallow
%load nnet_10_shallow.mat nnet_10_shallow
%load nnet_20_shallow.mat nnet_20_shallow
%save shallowWeights.mat weightsCell
% plot the error over time based on uvector vs output from nn

%save newNet128_64_16.m nnet
maxErr = zeros(4, 1);
avgErr = zeros(4, 1);
outputTest_3 = zeros(3, 1);
outputTest_5 = zeros(5, 1);
outputTest_10 = zeros(10, 1);
outputTest_20 = zeros(20, 1);
actual_3 = zeros(3, 1);
actual_5 = zeros(5, 1);
actual_10 = zeros(10, 1);
actual_20 = zeros(20, 1);
sqErr = zeros(4, 1);
%badGuess = zeros(numcell, 1);
%badinput = zeros(numcell, 1);
%actualsol = zeros(numcell, 1);

% split these for each network and output them
for i=(inputSize*.9+1):inputSize
    % plot it here
    outputTest_3 = nnet_3_shallow(inputdata_3(:,i));
    outputTest_5 = nnet_5_shallow(inputdata_5(:,i));
    outputTest_10 = nnet_10_shallow(inputdata_10(:,i));
    outputTest_20= nnet_20_shallow(inputdata_20(:,i));

    actual_3  = uVectors_3(:,i);
    actual_5 = uVectors_5(:,i);
    actual_10 = uVectors_10(:,i);
    actual_20 = uVectors_20(:,i);
    sqErr(1) = rms(actual_3 - outputTest_3);
    sqErr(2) = rms(actual_5 - outputTest_5);
    sqErr(3) = rms(actual_10- outputTest_10);
    sqErr(4) = rms(actual_20- outputTest_20);

    avgErr(1) = avgErr(1) + sqErr(1);
    avgErr(2) = avgErr(2) + sqErr(2);
    avgErr(3) = avgErr(3) + sqErr(3);
    avgErr(4) = avgErr(4) + sqErr(4);

end

output = nnet_3_shallow([0.8147, 0.9058, 0.1270]');
for k=1:4
    display("Avg Error: " + avgErr(k)/(inputSize*.1));
    display("Max Error: " + maxErr(k));
   %display(actual_3);
   %display(outputTest_3);
   %display(actual_5);
   %display(outputTest_5);
   %display(actual_10);
   %display(outputTest_10);
   %display(actual_20);
   %display(outputTest_20);
end
maxErr