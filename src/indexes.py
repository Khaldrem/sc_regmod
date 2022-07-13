import os
from src.utils import check_directory, check_working_os
from src.io import load_json, write_json


def insert_models_feature_importance(data = [], key="", phenotype="", exp_number = 0, index_dir_path = "", filename = ""):
    index_data = load_json(index_dir_path, filename)

    if key not in index_data.keys():
        index_data[key] = {}
    
    if "filter" not in index_data[key].keys():
        index_data[key]["filter"] = {}

    if phenotype not in index_data[key]["filter"].keys():
        index_data[key]["filter"][phenotype] = {}

    index_data[key]["filter"][phenotype][str(exp_number)] = data
    
    write_json(index_data, index_dir_path, filename)


def insert_length(length, key, index_dir_path, filename):
    index_data = load_json(index_dir_path, filename)

    if key not in index_data.keys():
        index_data[key] = {}

    index_data[key]["length"] = length

    write_json(index_data, index_dir_path, filename)


def insert_col_positions_data(columns_data=[], key="",index_dir_path = "", filename = ""):
    """
        Inserta en el archivo de indice, las posiciones
        de las columnas respecto al archivo

        Considerar la cadena procesos:
            original -> clean -> anova

        Entonces, si vemos el arreglo en el indice:
            {
                "clean":
                    "index": [1, 2, 3, 4]
            }
        
        Los valores en la lista refieren a las posiciones 
        en el archivo anterior, es decir, el original.
    """
    index_data = load_json(index_dir_path, filename)

    if key not in index_data.keys():
        index_data[key] = {}

    index_data[key]["index"] = columns_data
    
    write_json(index_data, index_dir_path, filename)


def create_index_file(path="", filename=""):
    """
        Crea el json de indice de un archivo,
        junto al dato inicial de su nombre
    """

    if path == "":
        print("path can't be empty.")
    
    if filename == "":
        print("filename can't be empty.")

    final_path = ""
    if check_working_os():
        final_path = f"{path}/{filename}.json"
    else:
        final_path = f"{path}\\{filename}.json"

    if not os.path.exists(final_path):
        check_directory(path)

        index = {
            "filename": filename
        }

        write_json(index, path, filename)
