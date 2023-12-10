% Open a file for writing
layerNum = 1;
nodeNum = 1;
bias = 0;
folder = ["network_3\", "network_5\", "network_10\", "network_20\"];

%load shallowWeights.mat
% correcting the orientation
for i=1:4
    weightsCell{i} = weightsCell{i}';
end

for k = 1:4
    for indexLayer = 1 : size(weightsCell{k}, 1)
        layerName = "Layer_" + layerNum;
        % alternate between weight and bias
        % first weight then bias
        nodeNum = 1;
        subMatrix = cell2mat(weightsCell{k}(indexLayer))'; 
        mSize = size(subMatrix,2);
        if size(subMatrix,1) == 1
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
            if size(subMatrix,1) == 1
                newFile = folder(k) + "Hidden" + layerName + "Bias_" + nodeName;
                layerNum = layerNum + 1;
            else
                newFile = folder(k) + "Connected" + layerName + "WeightNode_" + nodeName;
            end
            
            if bias == 0
                % write them normally
                writematrix(subMatrix(:, nodeNum), newFile);
            else
                % write the whole array since it's only 1 column
                writematrix(subMatrix(nodeNum, :), newFile);
            end
            nodeNum = nodeNum + 1;
            nodeName = "";
        end
    end
    LayerNum = 0;
    display(indexLayer);
end
