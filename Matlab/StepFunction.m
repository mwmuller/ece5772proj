
function [NextObs,Reward,IsDone,NextState] = StepFunction(Action,State, A, B)
    
    % system constraints 
    soc_lb = 0; soc_ub = 1;
    u_ub = 0.3;
    du_ub = 0.05; 
    threshold_end = 0.04; 

    % this is just the system dynamic equation 
    x = State; u = Action; 
    NextState = A*x + B*u; 
    NextObs = NextState; 
    
    % rewards
    r_time = -1; 
    %r_bal  = -1*norm(x - mean(x)); % maybe don't do this
    r_soc  = sum(-50.*(x > soc_ub | x < soc_lb)); % SOC is beyond or below bounds
    r_u    = sum(-50.*(abs(u) > u_ub));  % penalize if 
    %r_sign = 30.*all(u > 0) | all(u < 0);
    % r_du   = sum(-20.*(abs(delta_u) > du_ub)); % penalize if balancing currents change too harshly
    temp = abs(sum(u));
    if (temp < 0.03)
        temp = 0;
    end
    r_sum_bal = -100.*(temp);    % penalize if balancing currents don't sum to zero // adjust for .1% diff?
    
    %r_sum_bal_plus = 100.*(sum(u) > 0.99);
    if temp ~= 0 || r_soc ~= 0 || r_u ~= 0 % constraints 
        num_ub = sum(abs(u) > u_ub);
        r_u_reward = 0;
        if(num_ub > 0)
           u = (abs(u) - u_ub).*(abs(u) > u_ub);
           r_u_reward = -50.*(sum(u)/(num_ub*u_ub)); % vary the reward based on how far over we are
        end

        r_total = r_soc + r_u_reward + r_sum_bal; 
        Reward = r_total;
        IsDone = 1;
    else
        Reward = 0;
        if norm(x - mean(x)) < threshold_end % reached out goal
            IsDone = 1;
            Reward = 100;
        else
            IsDone = 0;
        end
    end

    