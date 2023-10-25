
function [NextObs,Reward,IsDone,NextState] = StepFuncDiscrete(Action,State, A)
    
    % system constraints 
    soc_lb = 0; soc_ub = 1;
    u_ub = 0.6;
    du_ub = 0.05; 
    threshold_end = 0.01;     
    diffCThresh = 0.20;
    r_diff = 0;
    tempC = (double(int32(rand(1)*3))+0.5)/100;
    % this is just the system dynamic equation 
    x = State; u = Action; % action selects the switch of the battery to be charged for 1 minute
    % u denotes the vectors which have they're switch active. (ONLY 1 at a
    % time)
    % Enable a switch to charge a battery
    % find the C max of each in pack cell lets assume 10000
    % change element u and add some random value

    if (sum(u) > 1)
        IsDone = 1;
        Reward = -100;
    end
    idx = find(u, 1);
    if (isempty(idx) == 1)
        % do nothing 
    else
        x(idx) = x(idx)+tempC;
        r_diff = -50.*(x(idx) ~= min(x) & ((max(x) - min(x)) > diffCThresh));
    end
    NextState = A*(x);
    NextObs = NextState; 
    % rewards
    r_time = -1;
    %r_bal  = -1*norm(x - mean(x)); % maybe don't do this
    r_soc  = sum((NextState > soc_ub | NextState < soc_lb)); % SOC is beyond or below bounds
    %Penalize if u is not the lowest SOC AND there is a difference greater
    %than thresh
    %otherwise it's fine
    IsDone = (sum(NextState > 0.94) == size(x)) | r_soc > 0;
    if (NextState > 0.9)
        %IsDone = 0;
    end
    IsDone = IsDone(1);
    if IsDone
        r_soc_safe = -100.*sum((NextState > soc_ub | NextState < soc_lb));
        %r_soc_under = -100.*sum(NextState < 0.96);
        %r_soc_correct = 100.*sum((NextState < 1) & (NextState > 0.93));
        %r_soc_correct = 0;
        Reward = r_soc_safe + r_time + r_diff;
    else % overcharged or not finished
        Reward = r_time + r_diff;
    end

    