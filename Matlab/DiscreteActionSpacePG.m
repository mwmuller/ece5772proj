% set the number of cells
num_cells = 3;

% initialize state transition matrices A and B 
A = eye(num_cells); 
B_cell = cell(1, num_cells);
for i = 1:num_cells
    B_cell{i} = (1/(3600*4.1)); 
end 
B = blkdiag(B_cell{:});

% define the state and action spaces 
ObsInfo = rlNumericSpec([num_cells 1]);
ObsInfo.Name = "Cell SOCs";
ActInfo = rlFiniteSetSpec({[0 0 0]', [1 0 0]', [0 1 0]', [0 0 1]'});
ActInfo.Name = "Cell Switches"; 

% Anonymous functions for reset and step functions
ResetHandler = @() ResetFuncDiscrete(num_cells); 
StepHandler = @(Action, Info) StepFuncDiscrete(Action, Info, A); 

% create the environment
env = rlFunctionEnv(ObsInfo, ActInfo, StepHandler, ResetHandler); 
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

rng(0);

% creating a policy gradient agent


actorNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(8)
    %reluLayer
    fullyConnectedLayer(16)
    fullyConnectedLayer(16)
    fullyConnectedLayer(numel(actInfo.Elements))
    ];


actorNet = dlnetwork(actorNet);
summary(actorNet)
%plot(actorNet)
myActor = rlDiscreteCategoricalActor(actorNet, obsInfo, actInfo);

agent = rlPGAgent(myActor);
%figure
%plot(actorNet)

%plot(layerGraph(actorNet))
%summary(actorNet)
%plot(layerGraph(criticNet))
%summary(criticNet)
prb = evaluate(myActor, {rand(obsInfo.Dimension)});
prb{1}

dist = evaluate(myActor, {rand(obsInfo.Dimension)});
dist{1}
agentOpts.ActorOptimizerOptions.LearnRate = 1e-2;
agentOpts.ActorOptimizerOptions.GradientThreshold = 0.9;

trainOpts = rlTrainingOptions(...
    MaxEpisodes=2000, ...
    MaxStepsPerEpisode=300, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=480,...
    ScoreAveragingWindowLength=100);


doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load("MATLABCartpolePG.mat","agent");
end

simOptions = rlSimulationOptions(MaxSteps=3000);
experience = sim(env,agent,simOptions);
totalReward = sum(experience.Reward)