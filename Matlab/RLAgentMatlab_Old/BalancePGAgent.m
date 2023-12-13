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
ActInfo = rlNumericSpec([num_cells 1]);
ActInfo.Name = "Balancing Currents [A]";

% Anonymous functions for reset and step functions
ResetHandler = @() ResetFunction(num_cells); 
StepHandler = @(Action, Info) StepFunction(Action, Info, A, B); 

% create the environment
env = rlFunctionEnv(ObsInfo, ActInfo, StepHandler, ResetHandler); 
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

rng(0);

% creating a policy gradient agent


inPath = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(10)
    fullyConnectedLayer(15)
    fullyConnectedLayer(prod(actInfo.Dimension), Name="inFC")
    ];

meanPath = [
    fullyConnectedLayer(prod(actInfo.Dimension), Name="meanOut") 
    ];

devPath = [
    softmaxLayer(Name="stdOut")
];

% connect layers
actorNet = layerGraph(inPath);
actorNet = addLayers(actorNet, meanPath);
actorNet = addLayers(actorNet, devPath);
actorNet = connectLayers(actorNet,"inFC","meanOut/in");
actorNet = connectLayers(actorNet,"inFC","stdOut/in");


%agent = rlPGAgent(obsInfo, actInfo);
%actorNet = getModel(getActor(agent));
%criticNet = getModel(getCritic(agent));
%myActor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ActionMeanOutputNames="meanOut", ActionStandardDeviationOutputNames="stdOut",ObservationInputNames="input");

initOpts = rlAgentInitializationOptions(NumHiddenUnit=128);

%agent = rlPGAgent(myActor);
agent = rlPGAgent(obsInfo,actInfo,initOpts);

agentOpts.ActorOptimizerOptions.LearnRate = 1e-3;
agentOpts.ActorOptimizerOptions.GradientThreshold = 1;


trainOpts = rlTrainingOptions(...
    MaxEpisodes=10000, ...
    MaxStepsPerEpisode=1000, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=10000,...
    ScoreAveragingWindowLength=100);

doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load("MATLABCartpolePG.mat","agent");
end

simOptions = rlSimulationOptions(MaxSteps=10000);
experience = sim(env,agent,simOptions);
totalReward = sum(experience.Reward)
save("AgentNet_10_30_Fresh_Maxed.mat", "agent");
%load("AgentNet_10_30_Fresh_Maxed.mat", "agent");