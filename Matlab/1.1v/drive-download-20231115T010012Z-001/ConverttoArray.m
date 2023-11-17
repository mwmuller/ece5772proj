% Open a file for writing
LayerNum = 1;
NodeNum = 1;
fileID = fopen('testing1.txt', 'a');
myMatrix = actorParams.Actor{1, 1};
for indexLayer = 0 : size(actorParams.Actor)
    layerName = "Layer_" + LayerNum;
    for NodeNum = 1 : size(myMatrix, 2)
        newFile = "Node_" + NodeNum;
        writematrix(myMatrix(:, LayerNum), )
    end
end
disp('Matrix elements written to output.txt');
writematrix(actorParams.Actor{1,1}(:, 1), "testing1.txt");
