import os


folder = open("network/")

# please get the folder names and use those to create the array. Each will be of size 
header = "nnArch.h" # the file to be created
hFile = open(header, 'a')
for file in folder:
    #get the name
    array = file.name()
    hFile.write("const double ")
    for line in file:
        # append things to the line

