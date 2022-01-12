from Bio import AlignIO
from os.path import isdir
import glob, sys, json

#Add valid extensions if the dataset changes
VALID_EXTENSIONS = [".phylip"]

#Todo escribir doc
def get_dataset_filepaths(base_dir="", extension=""):
    if(base_dir == ""):
        print("Arg 'base_dir' is empty.")
        return []
    
    if(extension == ""):
        print("Arg 'extension' is empty.")
        return []
    
    #Check if extension is valid
    if(extension not in VALID_EXTENSIONS):
        print("Not a valid extension.")
        return []

    #Get all files
    if(isdir(base_dir)):
        return glob.glob(f"{base_dir}/*{extension}")
    else:
        print("Arg 'base_dir' is not a directory.")
        return []


#TODO chequear entradas
def write_phylip_file(alignments, filepath="", filename=""):
    if(alignments != []):
        with open(f"{filepath}\\{filename}", "w") as handle:
            AlignIO.write(alignments, handle, "phylip-sequential")
    else:
        print(f"File: {filename} has no changes in it.")


def write_index_file(index, filepath="", filename=""):
    with open(f"{filepath}\\{filename}.json", "w") as outfile:
        json.dump(index, outfile)


#TODO Escribir doc
def read_phylip_file(file_path=""):
    if(file_path == ""):
        print("Arg 'base_dir' is empty.")
        sys.exit(1)

    return AlignIO.read(file_path, "phylip-relaxed")
