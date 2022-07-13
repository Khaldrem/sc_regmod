import pandas as pd
import os, glob, json
from sys import platform
from Bio import AlignIO

from src.utils import check_working_os, check_directory, get_anova_filepaths, get_filename
    

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
    # Checks if models/<type_model> folder exists
    # if not then create
    final_path = MODELS_BASE_PATH + "/" + type_model
    if not os.path.exists(final_path):
        os.mkdir(final_path)

    # Check if experiment folder exists
    final_path = final_path + "/" + exp_number
    if not os.path.exists(final_path):
        os.mkdir(final_path)
        os.mkdir(final_path + "/csv")

    if os.path.exists(final_path):
        files = glob.glob(f"{final_path}/*.joblib")
        if len(files) != 0:
            return True
    return False


def check_dataset_for_model_step(dataset_path, mode, exp_number):
    path = f"{dataset_path}/{mode}/{exp_number}"
    if os.path.exists(path):
        return True
    else:
        os.makedirs(path)
        return False


def get_id_values(example_fp):
    data = read_phylip_file(example_fp)
    id_rows = []
    for row in data:
        id_rows.append(row.id)
    
    return id_rows

def create_id_row_file(example_fp, models_path, exp_num):
    #Create if file not exists
    if not os.path.exists(f"{models_path}/id_rows.json"):
        f = open(f"{models_path}/id_rows.json", "w")
        json.dump({}, f)
        f.close()

    #Loads the json
    f = open(f"{models_path}/id_rows.json", "r")
    json_object = json.load(f)
    f.close()

    #Update id_rows
    id_rows_data = get_id_values(example_fp)

    json_object[exp_num] = id_rows_data

    #Writes it
    f = open(f"{models_path}/id_rows.json", "w")
    json.dump(json_object, f)
    f.close()


def create_file_data_csv(anova_path, models_path, exp_num):
    path = f"{models_path}/exp_{exp_num}_file_data.csv"

    #Checks if file doesnt exists
    if not os.path.exists(path):
        file_data = {
            "filename": [],
            "data_length": [],
            "filepath": []
        }

        filepaths = get_filepaths(anova_path)
        for f in filepaths:
            filename = get_filename(f)
            data = read_phylip_file(f)

            file_data["filename"].append(filename)
            file_data["data_length"].append(data.get_alignment_length())
            file_data["filepath"].append(f)

        file_data_df = pd.DataFrame.from_dict(file_data)
        file_data_df.to_csv(path)




    
