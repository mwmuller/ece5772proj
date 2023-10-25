% setting up environment/actor/observation and rng seed '0'
env = rlPredefinedEnv("CartPole-Discrete");

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

rng(0);

% creating a policy gradient agent


actorNet = [
    featureInputLayer(prod(obsInfo.Dimension(1)))
    fullyConnectedLayer(4)
    fullyConnectedLayer(8)
    fullyConnectedLayer(8)
    fullyConnectedLayer(numel(actInfo.Dimension(1)))
    ];

actorNet = dlnetwork(actorNet);
summary(actorNet);
plot(actorNet);

myActor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    ActionMeanOutputNames="fc_4",...
    ObservationInputNames="input");

prb = evaluate(myActor, {rand(obsInfo.Dimension)});
prb{1};

agent = rlPGAgent(myActor);


agent.AgentOptions.CriticOptimizerOptions.LearnRate = 5e-3;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;

trainOpts = rlTrainingOptions(...
    MaxEpisodes=1000, ...
    MaxStepsPerEpisode=500, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=480,...
    ScoreAveragingWindowLength=100);

plot(env);

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