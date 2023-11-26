import os

folder_path = 'network/'
output_path = 'nnArch.h'
  # Replace this with the path to your folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# List all files in the folder
files_in_folder = os.listdir(folder_path)

with open(output_path, 'a') as output:
    file_names = os.listdir(folder_path)
    # Iterate through the files and read their contents
    for file_name in files_in_folder:
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # Check if it's a file (not a subdirectory)
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
                elif "Bias" in file_name:
                    layerNum = parts[1][0]
                    nodeWeight = parts[2].split('.')[0]
                    arrayName = "Bias_L" + layerNum + "Node" + nodeWeight
                else:
                    layerNum = parts[0] # take the mean/std name
                    nodeWeight = parts[2].split('.')[0]
                    arrayName = layerNum + parts[1] +"_Node" + nodeWeight
                # display the correct layer/node and array
                #print("layerNum = " + layerNum + " | NodeWeight = " + nodeWeight)
                
                #print(arrayName)

                output.write("const double " + arrayName + "[] = {\n")
                commaCount = 1
                for line in file_contents:
                    # write to a file
                    line = line.rstrip('\n')
                    output.write(line)
                    commaCount = commaCount + 1 # increment to keep the comma count good
                    if commaCount < 257:
                        output.write(",")
                    output.write("\n")
                output.write("};\n") # finish the array brace and increment line
output.close()
                