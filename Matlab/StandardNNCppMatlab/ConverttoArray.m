% Open a file for writing

% DELETE THE FOLDER CONTENTS FIRST
% enter the folder names of the net you wish to extract
folder = ["network_3\", "network_5\", "network_10\", "network_20\"];


% load your net and CHECK THE NAMES
load net_20_cell_nopfcn.mat
load net_3_cell_nopfcn.mat
load net_5_cell_nopfcn.mat
load net_10_cell_nopfcn.mat


weightsCell = cell(4,1); % max size

% add the networks you want to extract here
netsCell = {net_3_cell, net_5_cell, net_10_cell, net_20_cell}; % match the folder/net order

for y=1:size(netsCell,2)
    tempnet = netsCell{y};

    if y < 4 % define if you need special extraction layers
        tempNetLearnables = {tempnet.IW{1}, tempnet.B{1}, tempnet.LW{2,1}, tempnet.B{2}, ...
            tempnet.LW{3,2}, tempnet.B{3}, tempnet.LW{4,3}, tempnet.B{4}, ...
            tempnet.LW{5,4}, tempnet.B{5}};
    else
        tempNetLearnables = {tempnet.IW{1}, tempnet.B{1}, tempnet.LW{2,1}, tempnet.B{2}, ...
            tempnet.LW{3,2}, tempnet.B{3}, tempnet.LW{4,3}, tempnet.B{4}, ...
            tempnet.LW{5,4}, tempnet.B{5}, tempnet.LW{6,5}, tempnet.B{6}};
    end
    % output size information for CPP usage
    tempnet.LW
    weightsCell{y} = tempNetLearnables';
end

for k = 1:size(netsCell,2)
    layerNum = 1;
    nodeNum = 1;
    bias = 0;
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
    display(layerNum);
end
