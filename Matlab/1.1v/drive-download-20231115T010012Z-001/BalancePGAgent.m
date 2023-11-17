
clear all
close all
clc

% set the number of cells
num_cells = 3;

% initialize state transition matrices A and B 
A = eye(num_cells); 
B_cell = cell(1, num_cells);
sampling_time = 10; 
for i = 1:num_cells
    B_cell{i} = (sampling_time/(3600*4.1)); 
end 
B = blkdiag(B_cell{:});

% define the state and action spaces 
ObsInfo = rlNumericSpec([num_cells 1]);
ObsInfo.Name = "Cell SOCs";
ActInfo = rlNumericSpec([num_cells 1]);
ActInfo.Name = "Balancing Currents [A]";

% Anonymous functions for reset and step functions
ResetHandler = @() ResetFunction(num_cells); 
StepHandler = @(Action, Info) StepFunction2(Action, Info, A, B); 

% create the environment
env = rlFunctionEnv(ObsInfo, ActInfo, StepHandler, ResetHandler); 
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

rng(0);

% creating a policy gradient agent


inPath = [
    featureInputLayer(prod(ObsInfo.Dimension), Name="obs_in")
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(256)
    reluLayer(Name="obs_out")
    ];

meanPath = [
    fullyConnectedLayer(prod(ActInfo.Dimension), Name="mean_path")
];

devPath = [
    fullyConnectedLayer(prod(ActInfo.Dimension), Name="std_path")
    softplusLayer(Name="std_out")
];

critic_net = [
    featureInputLayer(prod(ObsInfo.Dimension), Name="critic_inp")
    fullyConnectedLayer(50)
    leakyReluLayer
    fullyConnectedLayer(100)
    leakyReluLayer
    fullyConnectedLayer(100)
    leakyReluLayer
    fullyConnectedLayer(50)
    leakyReluLayer
    fullyConnectedLayer(1)
];


% connect actor layers
actorNet = layerGraph(inPath);
actorNet = addLayers(actorNet, meanPath);
actorNet = addLayers(actorNet, devPath);
actorNet = connectLayers(actorNet,"obs_out","std_path");
actorNet = connectLayers(actorNet,"obs_out","mean_path/in");

% create critic network
critic = rlValueFunction(dlnetwork(critic_net), obsInfo);

%agent = rlPGAgent(obsInfo, actInfo);
%actorNet = getModel(getActor(agent));
%criticNet = getModel(getCritic(agent));
%myActor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ActionMeanOutputNames="meanOut", ActionStandardDeviationOutputNames="stdOut",ObservationInputNames="input");

agentOpts = rlPGAgentOptions; 
agentOpts.UseBaseline = false; % "false" when no critic is used
agentOpts.ActorOptimizerOptions.LearnRate = 1e-2; % default
agentOpts.ActorOptimizerOptions.GradientThreshold = Inf; % default
agentOpts.SampleTime = sampling_time; 
actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    ActionMeanOutputNames="mean_path", ...
    ActionStandardDeviationOutputNames="std_out",...
    ObservationInputNames="obs_in");
agent = rlPGAgent(actor, agentOpts);
% agent = rlPGAgent(obsInfo, actInfo, agentOpts);
retrievedActorNet = layerGraph(getModel(getActor(agent)));
actorParams = getLearnableParameters(agent);
for()

trainOpts = rlTrainingOptions(...
    MaxEpisodes=20000, ...
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
    % Load the pretrained agent for the example and plot results. 
    % load("AgentNet_10_30_Fresh_Maxed.mat", "agent");
    simTime = 10000;
    samplingTime = 1; 
    numSteps = ceil(simTime/samplingTime);
    initialState = [0.4; 0.5; 0.6];
    states = zeros(num_cells, numSteps+1);
    currents = zeros(num_cells, numSteps+1);
    states(:,1) = initialState;

    for i=1:numSteps
        if i == 7000
            hi = 0;
        end 
        u = cell2mat(getAction(agent, states(:,i)));
        x_next = A*states(:,i) + B*u;
        states(:,i+1) = x_next;
        currents(:,i) = u; 
    end
    %%
    subplot(2,1,1)
    plot(1:numSteps, states(:,1:numSteps), 'LineWidth', 1.5)
    title('Cell States of Charge')
    xlabel('Time [s]')
    ylabel('SoC')
    legend('Cell 1', 'Cell 2', 'Cell 3')
    grid on

    subplot(2,1,2)
    plot(1:numSteps, currents(:,1:numSteps), 'LineWidth', 1.5)
    title('Cell Balancing Currents')
    xlabel('Time [s]')
    ylabel('Current [A]')
    legend('Cell 1', 'Cell 2', 'Cell 3')
    grid on
end

%%
simOptions = rlSimulationOptions(MaxSteps=10000);
experience = sim(env,agent,simOptions);
totalReward = sum(experience.Reward)
% save("AgentNet_10_30_Fresh_Maxed.mat", "agent");
