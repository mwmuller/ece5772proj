% Open a file for writing
layerNum = 1;
meanbool = 1;
inverting = 0;
nodeNum = 1;
folder = "network\";
myParams = actorParams.Actor;
for indexLayer = 1 : size(myParams)
    layerName = "Layer_" + layerNum;
    % alternate between weight and bias
    % first weight then bias
    nodeNum = 1;
    subMatrix = myParams{indexLayer, 1}; 
    mSize = size(subMatrix,2);
    if mSize == 256 && size(subMatrix, 1) == 3 % need to invert 
        inverting = 1;
    end
    for NodeNum = 1 : mSize
        if inverting == 1
            subMatrix = subMatrix';
        end
        if mSize == 1
            newFile = folder + "Hidden" + layerName + "Bias_" + nodeNum;
            layerNum = layerNum + 1;
        else
            newFile = folder + "Connected" + layerName + "WeightNode_" + nodeNum;
        end
        if inverting == 1 % need to invert 
            if meanbool == 1
                newFile = folder + "MeanOutput_" + "Weight_" + nodeNum;
                meanbool = 0;
            else
                newFile = folder + "StdDevOutput_" + "Weight_" + nodeNum;
            end
            subMatrix = subMatrix';
            inverting = 1;
        end
        writematrix(subMatrix(:, nodeNum), newFile);
        nodeNum = nodeNum + 1;
    end
    display(indexLayer);
    inverting = 0;
end
