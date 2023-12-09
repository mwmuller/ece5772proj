% Open a file for writing
layerNum = 1;
meanbool = 0;
inverting = 0;
nodeNum = 1;
stdbool = 0;
bias = 0;
folder = "network\";
myParams = actorParams.Actor;
for indexLayer = 1 : size(myParams)
    layerName = "Layer_" + layerNum;
    % alternate between weight and bias
    % first weight then bias
    nodeNum = 1;
    subMatrix = myParams{indexLayer, 1}; 
    mSize = size(subMatrix,1);
    if mSize == 3 && size(subMatrix, 2) == 256 % need to invert 
        inverting = 1;
        meanbool = ~meanbool;
        mSize = size(subMatrix,1);
    end
    if size(subMatrix,2) == 1
        % this be a bias matrix.
        bias = 1;
        mSize = 1;
    else
        bias = 0;
    end
    nodeName = "";
    for NodeNum = 1 : mSize
        if nodeNum < 100
            if nodeNum < 10
                nodeName = "0";
            end
            % append the nodenum
            nodeName = nodeName + "0" + nodeNum;
        else
            nodeName = nodeNum;
        end
        
        if inverting == 1
        end
        if size(subMatrix,2) == 1
            newFile = folder + "Hidden" + layerName + "Bias_" + nodeName;
            layerNum = layerNum + 1;
        else
            newFile = folder + "Connected" + layerName + "WeightNode_" + nodeName;
        end
        if inverting == 1 % need to invert 
            if meanbool == 1
                newFile = folder + "MeanOutput_" + "Weight_" + nodeName;
            end
            if stdbool == 1
                newFile = folder + "StdDevOutput_" + "Weight_" + nodeName;
            end
            inverting = 1;
        end
        
        if bias == 0
            % write them normally
            writematrix(subMatrix(nodeNum, :)', newFile);
        else
            % write the whole array since it's only 1 column
            writematrix(subMatrix(:, nodeNum), newFile);
        end
        nodeNum = nodeNum + 1;
        nodeName = "";
    end
    if meanbool == 1
        stdbool = 1;
    end
    display(indexLayer);

    inverting = 0;
end
