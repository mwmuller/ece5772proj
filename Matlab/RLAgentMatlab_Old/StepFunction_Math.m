function [NextObs,Reward,IsDone,NextState] = StepFunction_Math(Action,State, A, B)
    
    % system constraints 
    u_ub = 0.4;
    threshold_end = 0.03;
    a = 10;
    threshold_end_up = 0.35;
    % this is just the system dynamic equation 
    x = State; u = Action;
    NextState = A*x + B*u*10; 
    NextObs = NextState; 
   
    % rewards
    r_posMax = ~(find(x == max(x)) == find(u == min(u)));
    r_posMin = ~(find(x == min(x)) == find(u == max(u)));
    r_u    = sum(-50.*(abs(u) > u_ub));  % penalize if 
    %r_sign = 30.*all(u > 0) | all(u < 0);
    % r_du   = sum(-20.*(abs(delta_u) > du_ub)); % penalize if balancing currents change too harshly  
    if r_u ~= 0 || r_posMin ~= 0 || r_posMax ~= 0

        r_posMax = -100.*r_posMax;
        r_posMin = -100.*r_posMin;
        %r_balanced = -100.*r_balanced;
        r_total = r_u + r_posMax + r_posMin; 
        Reward = r_total;
        IsDone = 1; % termination Bad pick save time
        % [0, 0, 1] => balance to [0.333 0.333 0.333]
        % [0, 1, 1] => balance to [0.666 0.6666 0.6666]
    else % acceptable action zone
        % Now determine if it's efficient Or just barely passing
        % performing max time calculation
        xAvg = (sum(x)/length(x));
        avgDiff = (xAvg - x);
        maxDiff = max(abs(xAvg - x));
        count = sum(maxDiff == abs(avgDiff));
        maxTime = maxDiff/(B(1,1)*(u_ub/count));
    
        %simulateEnd = A*x' + B*u'*(maxTime*2);
        %normDiff = norm(simulateEnd - mean(simulateEnd));
        calcedU = (avgDiff / (maxTime*B(1,1)));
        if (u ~= calcedU)
            % not the rigt one
            r_badU = 1;
        end

        if r_badU ~= 0

            isDone = 1;
            r_badUVary = -10.*sum(abs(calcedU - u)./abs(u));
            r_total = sum(r_badUVary);% + r_posMax + r_posMin; 
            Reward = r_total;
        end
        Reward = -0.01; % better than bad actions, but still bad over time
        normDiff = norm(x - mean(x));
        if normDiff < threshold_end % reached out goal
            Reward = 10000; % must be comprable to the negative rewards to be meaningful
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
                tempExp = exp(-1*a*(normDiff-threshold_end));
            else
                % do not change reward from -1
            end
            Reward = Reward + tempExp;
            IsDone = 0;
        end
    end

    