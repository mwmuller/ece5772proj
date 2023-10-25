
function [InitialObservation, InitialState] ...
    = ResetFunction(num_cells)

% we initialize the cells' SOC's symmetrically around 
% a desired level -- let's arbitrarily select 0.5 as said desired level,
% and let's have the cell's SOC's distributed from 0.4 to 0.6

InitialState = linspace(0.4, 0.6, num_cells)'; 
InitialObservation = InitialState; 

end 