
% set the number of cells
num_cells = 3;

% initialize state transition matrices A and B 
A = eye(num_cells); 
B_cell = cell(1, num_cells);
for i = 1:num_cells
    B_cell{i} = 1/(3600*4.1); 
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

% create the agent ... Michael's stuff goes here
