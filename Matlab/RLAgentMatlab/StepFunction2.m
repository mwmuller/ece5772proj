
function [NextObs,Reward,IsDone,NextState] = StepFunction2(Action,State, A, B)
    
    % system constraints 
    soc_lb = 0; soc_ub = 1;
    u_ub = 0.5;
    du_ub = 0.05; 
    threshold_end = 0.01; 

    % this is just the system dynamic equation 
    x = State; u = Action; 
    NextState = A*x + B*u; 
    NextObs = NextState; 
    
    % rewards
    r_time = -1; 
    r_bal  = -1*norm(x - mean(x)); % maybe don't do this
    r_soc  = sum(-50.*(x > soc_ub | x < soc_lb)); % SOC is beyond or below bounds
    r_u    = sum(-50.*(abs(u) > u_ub));  % penalize if 
    %r_sign = 30.*all(u > 0) | all(u < 0);
    % r_du   = sum(-20.*(abs(delta_u) > du_ub)); % penalize if balancing currents change too harshly
    r_sum_bal = -100.*(sum(u) ~= 0);    % penalize if balancing currents don't sum to zero
    r_total = r_time + r_soc + r_u + r_bal + r_sum_bal; 
    Reward = r_total;
    if norm(x - mean(x)) < threshold_end
        IsDone = 1;
    else
        IsDone = 0;
    end
    