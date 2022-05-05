from multiprocessing import Pool, cpu_count
from functools import partial

import time
from datetime import datetime
from random import randint

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO

import scipy.stats as stats
import os, sys
import numpy as np
import pandas as pd

from src.io import *
from src.utils import *
from src.sequences import create_compressed_alignment

INDEXES_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\indexes"
PHENOTYPE_FILEPATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\fenotipos\\fenotipos_clean.csv"
COMPRESSED_DATASET_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\compressed_sequences"
ANOVA_DATASET_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\anova"
ANOVA_RES_DATASET_PATH = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\anova_res"


def detect_pvalue(pvalue):
    if pvalue <= 0.05:
        return True
    return False


def do_anova(phenotypes_dataset, phenotypes_names, data_ids, data):
    # print(f"Proceso con archivo: {data['filename']} - {pheotypes_names} - {data_ids}")
    
    # time_sleep = randint(1,10)
    # time.sleep(time_sleep)
    # print(f"Proceso con archivo: {data['filename']} - Duerme: {time_sleep}")
    # print(f"Proceso con archivo: {data['filename']} - Finalizado")

    start = time.time()
    n = len(data["data"])

    col_n = data["data"].get_alignment_length()
    save_anova_res = {
        'col': range(col_n),
        'SM300-Efficiency': np.zeros(col_n),
        'SM300-Rate': np.zeros(col_n),
        'SM300-Lag': np.zeros(col_n),
        'SM300-AUC': np.zeros(col_n),
        'SM60-Efficiency': np.zeros(col_n),
        'SM60-Rate': np.zeros(col_n),
        'SM60-Lag': np.zeros(col_n),
        'SM60-AUC': np.zeros(col_n),
        'Ratio-Efficiency': np.zeros(col_n),
        'Ratio-Rate': np.zeros(col_n),
        'Ratio-Lag': np.zeros(col_n),
        'Ratio-AUC': np.zeros(col_n)
    }
    
    anova_res_per_phenotype_df = pd.DataFrame(data=save_anova_res)
    cols_not_eliminated = []

    for col in range(data["data"].get_alignment_length()):
        #print(f"col: {col}")
        
        df_structure = {
            'ids': data_ids, 
            'base': list(data["data"][:, col]),
            'SM300-Efficiency': np.zeros(n),
            'SM300-Rate': np.zeros(n),
            'SM300-Lag': np.zeros(n),
            'SM300-AUC': np.zeros(n),
            'SM60-Efficiency': np.zeros(n),
            'SM60-Rate': np.zeros(n),
            'SM60-Lag': np.zeros(n),
            'SM60-AUC': np.zeros(n),
            'Ratio-Efficiency': np.zeros(n),
            'Ratio-Rate': np.zeros(n),
            'Ratio-Lag': np.zeros(n),
            'Ratio-AUC': np.zeros(n)
        }

        col_df = pd.DataFrame(data=df_structure)

        #Añado los datos del csv de fenotipos al dataframe 
        for index, row in col_df.iterrows():
            filtered = phenotypes_dataset.loc[phenotypes_dataset["Standard"] == row["ids"]]
            filtered = filtered.values[:, 2:14].tolist()
            
            if len(filtered) != 0:
                col_df.loc[index, ["SM300-Efficiency", "SM300-Rate","SM300-Lag","SM300-AUC",
                                "SM60-Efficiency","SM60-Rate","SM60-Lag","SM60-AUC",
                                "Ratio-Efficiency","Ratio-Rate","Ratio-Lag","Ratio-AUC"]] = filtered[0]

        #Obtengo las llaves unicas para esa columna
        unique_keys = set(list(data["data"][:, col]))

        #Ahora por cada fenotipo, realizo el anova de esta columna
        #primero creo un dict de cada fenotipo
        col_anova_data = {}
        for phenotype in phenotypes_names:
            col_anova_data[phenotype] = {}
            
            for key in unique_keys:
                col_anova_data[phenotype][key] = col_df.loc[col_df["base"] == key, phenotype].tolist()

        #Se utiliza para saber si la columna se conserva en el archivo o se elimina
        delete_col = True
        phenotypes_test_results = []
        
        # Ahora que tengo el dataframe con los datos de cada fenotipo por base de la columna
        # puedo realizar el test anova por cada fenotipo
        df_to_test = pd.DataFrame.from_dict(col_anova_data, orient='index')
        
        if len(df_to_test.loc[phenotype]) != 1:
            for phenotype in phenotypes_names:
                res = stats.f_oneway(*df_to_test.loc[phenotype])
                #print(f"col: {col} - fenotipo: {phenotype}")
                #print(f"    pvalue: {res.pvalue}")
                
                #Detectamos el pvalue para este fenotipo
                #si es positivo quiere decir que la columna se conserva, sino se elimina
                if detect_pvalue(res.pvalue):
                    phenotypes_test_results.append(1)
                    delete_col = False
                else:
                    phenotypes_test_results.append(0)
            
            #Añado los resultados obtenidos de los anovas al df
            anova_res_per_phenotype_df.loc[col, ["SM300-Efficiency", "SM300-Rate","SM300-Lag","SM300-AUC",
                                                "SM60-Efficiency","SM60-Rate","SM60-Lag","SM60-AUC",
                                                "Ratio-Efficiency","Ratio-Rate","Ratio-Lag","Ratio-AUC"]] = phenotypes_test_results
        
        if not delete_col:
            cols_not_eliminated.append(col)


    #Write anova res
    anova_res_per_phenotype_df.to_csv(f"{ANOVA_RES_DATASET_PATH}\\{data['filename']}.csv")

    data["new_data"] = create_compressed_alignment(data["data"], cols_not_eliminated)
    
    if cols_not_eliminated != []:
        #Write file
        write_phylip_file(data["new_data"], ANOVA_DATASET_PATH, data["filename"])
    else:
        #Write file
        write_phylip_file(data["data"], ANOVA_DATASET_PATH, data["filename"])
    
    #Update index
    update_index_file_anova(cols_not_eliminated, INDEXES_PATH, data["filename"])
    
    end = time.time()
    print(f"File: {data['filename']} / data: {data['data'].get_alignment_length()} / new_data: {len(cols_not_eliminated)} took: {end-start}")



def get_data_ids(dataset):
    #Todos los archivos poseen los mismos IDS
    #en el mismo orden, por lo que extraemos los del 1er. archivo
    ids = []
    for elem in dataset[0]["data"]:
        ids.append(elem.id)

    return ids


def clean_dataset_by_id(dataset, phenotypes_dataset):
    phenotypes_ids = phenotypes_dataset.loc[:, "Standard"].tolist()

    for data in dataset:
        new_seq = []
        for row in data["data"]:
            if row.id in phenotypes_ids:
                new_seq.append(row)

        data["data"] = MultipleSeqAlignment(new_seq)
    
    return dataset


def check_anova_file(filename="", base_dir=""):
    if os.path.exists(f"{base_dir}/{filename}.phylip"):
        return True
    
    return False


if __name__ == "__main__":
    #Get filepaths
    filepaths = get_filepaths(COMPRESSED_DATASET_PATH)

    #Cargamos el dataset de fenotipos
    phenotypes_dataset = read_phenotypes_file(PHENOTYPE_FILEPATH)

    max_files = 200
    is_finished = False

    while not is_finished:
        temp = 0
        #Cargamos en memoria el dataset
        dataset = []
        for filepath in filepaths:
            #Chequeo si ya fue testeado
            filename = get_filename(filepath=filepath)
            if not check_anova_file(filename, ANOVA_DATASET_PATH) and temp < max_files:
                #print(f"Nuevo archivo: {filename}")
                data = {}
                data["filepath"] = filepath
                data["filename"] = filename
                data["data"] = read_phylip_file(filepath)

                dataset.append(data)
                temp += 1
        
        if dataset == []:
            is_finished = True
        else:
            #Limpiar datos, debido a que el archivo de fenotipos no
            #presenta todos los ID, debemos remover aquellas columnas que no se encuentran.
            #dataset = clean_dataset_by_id(dataset, phenotypes_dataset)
             
            dataset = clean_dataset_by_id(dataset, phenotypes_dataset)

            print("===== INICIANDO ======")
            print("fecha: ", datetime.today().strftime('%d-%m-%Y %H:%M:%S'))
            print("======================")

            POOL_WORKERS = 16

            if dataset != []:
                phenotypes_names = list(phenotypes_dataset.keys()[2:-1].astype('str').values)
                data_ids = get_data_ids(dataset)
                
                start = time.time()
                
                #Iniciar proceso por cada archivo
                p = Pool(POOL_WORKERS, maxtasksperchild=1)
                p.map(partial(do_anova,phenotypes_dataset, phenotypes_names, data_ids) , dataset)
                
                end = time.time()
                print("Total:", end-start)
            else:
                print("dataset is empty.")
