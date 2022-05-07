import pandas as pd
import os, glob, json
from sys import platform
from Bio import AlignIO


def get_filepaths(base_path = ""):
    if base_path == "" or base_path is None:
        print("Variable 'base_path' cannot be empty or None")
        return []
    
    #Get all files
    if(os.path.isdir(base_path)):
        return glob.glob(f"{base_path}/*.phylip")
    else:
        print("Arg 'base_dir' is not a directory.")
        return []


def read_phylip_file(filepath=""):
    if filepath == "":
        print("Arg 'filepath' is empty.")
        return []
    
    return AlignIO.read(filepath, "phylip-relaxed")


def read_phenotypes_file(filepath=""):
    return pd.read_csv(filepath, sep=",")


def write_phylip_file(data, path="", filename=""):
    if data == [] or data is None:
        print("data cannot be empty.")
        return False
    
    if path == "" or path is None:
        print("path cannot be empty.")
        return False
    
    if filename == "" or filename is None:
        print("filename cannot be empty.")
        return False
    
    final_path = ""
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        final_path = f"{path}/{filename}.phylip"
    
    if platform == "win32":
        final_path = f"{path}\\{filename}.phylip"
        
    with open(final_path, "w") as handle:
        AlignIO.write(data, handle, "phylip-sequential")
        
    return True


def write_compressed_index(data, filepath = ""):
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile)


def check_index_file(filename = "", indexes_path = ""):
    #Checks if 'indexes' folder exists
    if not os.path.exists(indexes_path):
        os.makedirs(indexes_path)

    #Checks if 'filename' exists, if not it creates it
    if not os.path.exists(f"{indexes_path}/{filename}"):
        return False

    return True


def update_index_file_anova(data, json_filepath, filename):
    #Load index
    f = open(f"{json_filepath}\\{filename}.json", "r")
    json_object = json.load(f)
    f.close()

    original_positions = []
    for el in data:
        original_positions.append(json_object["compressed_pos"][el])

    #Update
    json_object["anova_pos"] = original_positions
    
    #Write updated version
    f = open(f"{json_filepath}\\{filename}.json", "w")
    json.dump(json_object, f)
    f.close()


def update_index_base_data_anova(data, json_filepath, filename):
    #Load index
    f = open(f"{json_filepath}/{filename}.json", "r")
    json_object = json.load(f)
    f.close()

    json_object["anova_data"] = {}
    json_object["anova_data"]["bases"] = data

    #Write updated version
    f = open(f"{json_filepath}/{filename}.json", "w")
    json.dump(json_object, f)
    f.close()

    