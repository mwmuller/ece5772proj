import os
import sys

# CHANGE FOR A SPECIFIC NETWORK EXTRACTION
netSizeArr = ['3', '5', '10', '20']

for netSize in netSizeArr:
    folder_path = 'network_' + netSize + '/'
    output_weight = 'nnArchWeights_' + netSize + '.h'
    output_bias = 'nnArchBias_' + netSize + '.h'
    fileSelect = 0 # 0 = weight by default | 1 = bias file
    # Replace this with the path to your folder
    mypath = os.path.dirname(os.path.abspath(sys.argv[0])) + "/"

    finalPath = mypath + folder_path
    # List all files in the folder
    files_in_folder = os.listdir(finalPath)
    files_in_folder = sorted(files_in_folder)

    if os.path.isfile(mypath + output_weight):
        os.remove(mypath + output_weight)
    if os.path.isfile(mypath + output_bias):
        os.remove(mypath + output_bias)
    with open(mypath + output_weight, 'a') as o_weight:
        with open(mypath + output_bias, 'a') as o_bias:

            file_names = os.listdir(finalPath)
            # Iterate through the files and read their contents
            o_weight.write("const double network_weights_"+ netSize +"[] = {\n")
            o_bias.write("const double network_biases_"+ netSize +"[] = {\n")
            for file_name in files_in_folder:
                print(file_name)
                file_path = os.path.join(mypath, folder_path, file_name)
                #if os.path.isfile(file_path):  # Check if it's a file (not a subdirectory)
                with open(file_path, 'r') as file:

                    parts = file_name.split('_')
                    # [1] first elements contains the layer
                    # [2] second element contains the node weights

                    file_contents = file.readlines()
                    arrayName = ""
                    if "Connected" in file_name:
                        layerNum = parts[1][0]
                        nodeWeight = parts[2].split('.')[0]
                        arrayName = "Weights_L" + layerNum + "Node" + nodeWeight
                        fileSelect = 0
                    elif "Bias" in file_name:
                        layerNum = parts[1][0]
                        nodeWeight = parts[2].split('.')[0]
                        arrayName = "Bias_L" + layerNum + "Node" + nodeWeight
                        fileSelect = 1
                    else:
                        layerNum = parts[0] # take the mean/std name
                        nodeWeight = parts[2].split('.')[0]
                        arrayName = layerNum + parts[1] +"_Node" + nodeWeight
                        fileSelect = 0
                    # display the correct layer/node and array
                    #print("layerNum = " + layerNum + " | NodeWeight = " + nodeWeight)
                    
                    #print(arrayName)
                    if fileSelect == 0:
                        fileOut = o_weight
                    elif fileSelect == 1:
                        fileOut = o_bias
                    # iterate through the contents
                    fileOut.write("\n // Description of following elements " + arrayName + "\n")   
                    for line in file_contents:
                        # write to a file
                        line = line.rstrip('\n')
                        fileOut.write(line)
                        fileOut.write(",") # remove the last one manually, cuz I'm lazy
                        fileOut.write("\n") 
            o_bias.write("};\n") # finish the array brace and increment line
        o_weight.write("};\n") # finish the array brace and increment line
    o_weight.close()
    o_bias.close()

print("DONT FORGET TO REMOVE THE ',' FROM THE END OF EACH ARRAY!!!")
                