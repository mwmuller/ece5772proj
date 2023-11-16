
function [NextObs,Reward,IsDone,NextState] = StepFunction2(Action,State, A, B)
    
    % system constraints 
    soc_lb = 0; soc_ub = 1;
    u_ub = 0.4;
    %du_ub = 0.05;
    a = 100;
    threshold_end = 0.03;
    threshold_end_up = 0.50;
    % this is just the system dynamic equation 
    x = State; u = Action; 
    NextState = A*x + B*u; 
    NextObs = NextState; 
    
    % rewards
    r_time = -1; % this was positive 1, is that intended? 
    %r_bal  = -1*norm(x - mean(x)); % maybe don't do this
    r_soc  = sum(-500.*(x > soc_ub | x < soc_lb)); % SOC is beyond or below bounds
    r_u    = sum(-50.*(abs(u) > u_ub));  % penalize if 
    %r_sign = 30.*all(u > 0) | all(u < 0);
    % r_du   = sum(-20.*(abs(delta_u) > du_ub)); % penalize if balancing currents change too harshly
    temp = abs(sum(u));
    if (temp < 0.03)
        temp = 0;
    end
    r_sum_bal = -400.*(temp);    % penalize if balancing currents don't sum to zero // adjust for .1% diff?
    
    %r_sum_bal_plus = 100.*(sum(u) > 0.99);
    % Teach agent to abide by constraints
    % large punishments
    % greatest value should have a - current
    % lowest value should have highest current]

    r_posMax = ~(find(x == max(x)) == find(u == min(u)));
    r_posMin = ~(find(x == min(x)) == find(u == max(u)));
  %if(~isempty(find(u == 0)))
  %    r_balanced = (find(x == (sum(x)/3)) == find(u == 0)); % 0 current at balanced index
  %else
  %    %% check if any are balanced and we don't have 0 in u
  %    r_balanced = find(x == (sum(x)/3));
  %    if(isempty(r_balanced))
  %        r_balanced = 0;
  %    end
  %end
    %avgSOC = sum(x)/3; % get avg soc and determine if a cell doesn't need charge/discharge
    %myAvg = x((x == avgSOC));
    
    Reward = 0;
    %if temp ~= 0 || r_soc ~= 0 || r_u ~= 0 || r_posMin ~= 0 || r_posMax ~= 0 %|| r_balanced ~=0
     if temp ~= 0 || r_soc ~= 0 || r_u ~= 0 
        num_ub = sum(abs(u) > u_ub); % get number of instances 
        r_u_reward = 0;
        if(num_ub > 0)
           u = (abs(u) - u_ub).*(abs(u) > u_ub);
           % u => saturate all instances that are within the bounds (0
           % means within bounds)
           r_u_reward = -1000.*(sum(u)/(num_ub*u_ub)); % vary the reward based on how far over we are
           % num_ub = instances where u exceedes u_ub
           % num_ub*u_ub = total u_ub current amount (u_ub = 0.3 -> 0.6 for
           % 2 istances of excess)
           % r_u_reward what % over u_ub_total is the action? Multiply that
           % % by -100 to get a scalable reward
        end
        % r_posMax = -100.*r_posMax;
        % r_posMin = -100.*r_posMin;
        %r_balanced = -100.*r_balanced;
        % r_total = r_soc + r_u_reward + r_sum_bal + r_posMax + r_posMin + r_time; 
        r_total = r_soc + r_u_reward + r_sum_bal + r_time; 
        Reward = r_total;
        IsDone = 1; % termination Bad pick save time
        % [0, 0, 1] => balance to [0.333 0.333 0.333]
        % [0, 1, 1] => balance to [0.666 0.6666 0.6666]
     end

    % acceptable action zone
    % originally -0.01, better than bad actions, but still bad over time
    normDiff = rms(x - mean(x)); % this is more scalable, the norm gets larger with more cells
    if normDiff < threshold_end % reached our goal
        Reward = Reward + 10000; % must be comprable to the negative rewards to be meaningful
        IsDone = 1;
        % compete with r_time*numberofsteps taken
        % prevents action from not charging and simply meeting the
        % conditions to stay in the else statement
        % the vector [0 -0.1 0.1] would work but isn't correct
    else % The closer you get over time, the btter the reward
        % determine what normdiff is in respect to threshold_end
        % reward for being close
        % punish for being too far but also an acceptable action
        tempExp = 0;
        if(normDiff < threshold_end_up) % are we close enough to start issueing a reward
            tempExp = 10000*exp(-1*a*(normDiff-threshold_end));
        else
            % do not change reward from -1
        end
        Reward = Reward + tempExp;
        IsDone = 0;
    end
end

    