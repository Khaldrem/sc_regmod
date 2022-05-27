import pandas as pd
import os, glob, json
from sys import platform
from Bio import AlignIO

from src.utils import check_working_os, check_directory, get_anova_filepaths
    

def get_filepaths(base_path = ""):
    if base_path == "" or base_path is None:
        print("Variable 'base_path' cannot be empty or None")
        return []

    if(os.path.isdir(base_path)):
        return glob.glob(f"{base_path}/*.phylip")

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
    if path == "" or path is None:
        print("path cannot be empty.")
        return False
    
    if filename == "" or filename is None:
        print("filename cannot be empty.")
        return False
    
    if data == [] or data is None:
        print(f"file: {filename} have empty data.")
        return False

    #Check if directory exists, if not create one
    check_directory(path)


    final_path = ""
    if check_working_os():
        final_path = f"{path}/{filename}.phylip"
    else:
        final_path = f"{path}\\{filename}.phylip"
        
    with open(final_path, "w") as handle:
        AlignIO.write(data, handle, "phylip-sequential")
        
    return True


def write_pandas_csv(data, path="", filename=""):
    final_path = ""
    check_directory(path)

    if check_working_os():
        final_path = f"{path}/{filename}.csv"
    else:
        final_path = f"{path}\\{filename}.csv"

    data.to_csv(final_path)


def load_json(path="", filename=""):
    if path == "" or path is None:
        print("path cannot be empty.")
        return False
    
    if filename == "" or filename is None:
        print("filename cannot be empty.")
        return False

    check_directory(path)

    final_path = ""
    if check_working_os():
        final_path = f"{path}/{filename}.json"
    else:
        final_path = f"{path}\\{filename}.json"

    f = open(final_path, "r")
    json_object = json.load(f)
    f.close()

    return json_object


def write_json(data, path = "", filename = ""):
    if path == "" or path is None:
        print("path cannot be empty.")
        return False
    
    if filename == "" or filename is None:
        print("filename cannot be empty.")
        return False

    check_directory(path)

    final_path = ""
    if check_working_os():
        final_path = f"{path}/{filename}.json"
    else:
        final_path = f"{path}\\{filename}.json"

    with open(final_path, 'w') as outfile:
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
    f = open(f"{json_filepath}/{filename}.json", "r")
    json_object = json.load(f)
    f.close()

    original_positions = []
    for el in data:
        original_positions.append(json_object["compressed_pos"][el])

    #Update
    json_object["anova_pos"] = original_positions
    
    #Write updated version
    f = open(f"{json_filepath}/{filename}.json", "w")
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


def check_if_anova_files_exists(base_path="", type_anova="", p_value_threshold=0, chromosome=""):
    if base_path == "" or type_anova == "" or chromosome == "":
        print("A parameter in check_anova_files is empty.")
        return False
    
    anova_path = get_anova_filepaths(base_path, type_anova, p_value_threshold, chromosome)
    if anova_path != "":
        files = get_filepaths(anova_path)
        if len(files) != 0:
            return True
            
    return False


def check_if_models_exists(exp_number = "", MODELS_BASE_PATH="", type_model=""):
    final_path = MODELS_BASE_PATH + "/" + type_model + "/" + exp_number
    if os.path.exists(final_path):
        files = glob.glob(f"{final_path}/*.joblib")
        if len(files) != 0:
            return True
    return False