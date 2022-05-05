from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO

import scipy.stats as stats
import os, sys, time
import numpy as np
import pandas as pd
from src.io import read_phenotypes_file, read_phylip_file
from .utils import get_filename


def detect_pvalue(pvalue):
    if pvalue <= 0.05:
        return True
    return False


def get_data_ids(dataset):
    #Todos los archivos poseen los mismos IDS
    #en el mismo orden, por lo que extraemos los del 1er. archivo
    ids = []
    for elem in dataset[0]["data"]:
        ids.append(elem.id)

    return ids


def do_anova_task(dataset, phenotypes_dataset):
    #Extrae los nombres de los fenotipos
    phenotypes_names = list(phenotypes_dataset.keys()[2:-1].astype('str').values)
    #phenotypes_standard = phenotypes_dataset.loc[:, "Standard"].tolist()

    """
        Por cada archivo. 
        data = {
            "filepath": './etc/',
            "filename": 'YALSAD...',
            "data": MultipleSeqAlignment object
        }
    """

    data_ids = get_data_ids(dataset)

    for data in dataset:
        start = time.time()

        #print("===========")
        #print(f"len: {data['data'].get_alignment_length()}")

        n = len(data["data"])

        #Almacena los resultados del anova por cada columna en forma de 1 o 0
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

        #Por cada columna en los datos
        for col in range(data["data"].get_alignment_length()):
            start2 = time.time()
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

            #Ahora que tengo el dataframe con los datos de cada fenotipo por base de la columna
            #puedo realizar el test anova por cada fenotipo
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
            #else:
            #    print(f"col: {col} is eliminated")


            #print(f"len(phenotypes_test_results): {len(phenotypes_test_results)}")

            
            #print(f"cols not eliminated: {cols_not_eliminated}")
            #print(f"phenotypes_test_results: {phenotypes_test_results}")
            #print(anova_res_per_phenotype_df.head(n=30))
            
            end2 = time.time()
            print("Time per col :", end2-start2)
        end = time.time()
        print("Total by file:", end-start)
        #sys.exit()
        break

        #data["cols_not_eliminated"] = cols_not_eliminated
        #print(data["cols_not_eliminated"])

        



            # data_to_test = {}

            # #Por cada fenotipo
            # for phenotype in phenotypes_names:
            #     print(f"     {phenotype}")
            #     data_to_test[phenotype] = {}
            #     for key in unique_keys:
            #         data_to_test[phenotype][key] = []

            #     #Por cada fila en la columna
            #     for row in range(len(data["data"])):
            #         if data["data"][row].id in phenotypes_standard:
            #             base = data["data"][row, col]
            #             datum = phenotypes_dataset[phenotypes_dataset["Standard"] == data["data"][row].id][phenotype].astype('float64').values[0]
            #             data_to_test[phenotype][base].append(datum)

            # anova_data.append(data_to_test)

        
            # print()
            # print(anova_data[0])
            # print("===========")

            



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
    if os.path.exists(f"{base_dir}/{filename}.fasta.phylip"):
        return True
    
    return False


def do_anova(filepaths=[], base_dir="", phenotypes_filepath=""):
    if filepaths == []:
        print("filepaths is empty")
        return False

    if base_dir == "":
        print("base_dir is empty")
        return False

    if phenotypes_filepath == "":
        print("phenotypes_filepath is empty")
        return False

    #Cargamos el dataset de fenotipos
    phenotypes_dataset = read_phenotypes_file(phenotypes_filepath)

    #Cargamos en memoria el dataset
    dataset = []
    for filepath in filepaths:
        #Chequeo si ya fue testeado
        filename = get_filename(filepath=filepath)
        if not check_anova_file(filename, base_dir):
            #print(f"Nuevo archivo: {filename}")
            data = {}
            data["filepath"] = filepath
            data["filename"] = filename
            data["data"] = read_phylip_file(filepath)

            dataset.append(data)


    print(f"filename: {dataset[0]['filename']} / len: {len(dataset[0]['data'])}")
    print(f"filename: {dataset[10]['filename']} / len: {len(dataset[10]['data'])}")

    #Limpiar datos, debido a que el archivo de fenotipos no
    #presenta todos los ID, debemos remover aquellas columnas que no se encuentran.
    #dataset = clean_dataset_by_id(dataset, phenotypes_dataset)
    dataset = clean_dataset_by_id(dataset, phenotypes_dataset)

    print(f"filename: {dataset[0]['filename']} / len: {len(dataset[0]['data'])}")
    print(f"filename: {dataset[10]['filename']} / len: {len(dataset[10]['data'])}")

    #Por cada archivo realizamos el anova correspondiente
    do_anova_task(dataset, phenotypes_dataset)




# def clean_compressed_file(path, standard_list):
#     data = read_phylip_file(path)
    
#     new_align = []
#     for row in range(len(data)):
#         if data[row].id in standard_list:
#             new_align.append(SeqRecord(Seq(data[row].seq), id=data[row].id))
#     return MultipleSeqAlignment(new_align)


# def get_index_based_on_id(data, seq_id):
#     index = 0
#     for row in range(len(data)):
#         if seq_id == data[row].id:
#             #print(f"seq_id: {seq_id} , data[row].id: {data[row].id}")
#             return index
#         index += 1
#     return -1

# def prepare_dataframe(data, phenotype_df, type_haploide_diploide):
#     """
#         - type_phenotype: Posee 4 tipos de entrada que corresponden con las columnas de los fenotipos -> "Efficiency", "Rate", "Lag", "AUC"
#         - type_haploide_diploide: 4 tipos de entrada, "all", "haploide_euploide", "diploide_euploide", "--"
#     """
    
#     """
#     if type_phenotype == "Efficiency":
#         new_dataframe = phenotype_df[["Standard", "SM300-Efficiency", "SM60-Efficiency", "Ratio-Efficiency"]]
#     elif type_phenotype == "Rate":
#         new_dataframe = phenotype_df[["Standard", "SM300-Rate", "SM60-Rate", "Ratio-Rate"]]
#     elif type_phenotype == "Lag":
#         new_dataframe = phenotype_df[["Standard", "SM300-Lag", "SM60-Lag", "Ratio-Lag"]]
#     elif type_phenotype == "AUC":
#         new_dataframe = phenotype_df[["Standard", "SM300-AUC", "SM60-AUC", "Ratio-AUC"]]
#     else:
#         print("Wrong 'type_phenotype'.")
#         return []
#     """
    
    
#     if type_haploide_diploide == "all":
#         new_dataframe = phenotype_df.copy()
        
#         new_columns = {}
#         for col in range(data.get_alignment_length()):
#             col_seq = []
#             for ind in new_dataframe.index:
#                 index = get_index_based_on_id(data, new_dataframe["Standard"][ind])
#                 col_seq.append(data[index, col])

#             new_key = "x" + str(col)
#             new_columns[new_key] = pd.Series(col_seq, index=new_dataframe.index, name=new_key)
        
#         temp = pd.DataFrame(new_columns)
#         final = pd.concat([new_dataframe, temp], axis=1)
#         return final
        
#     else:
#         print("Not implemented yet.")
#         return []    
    
    
# def get_phenotype_data_per_base(data, data_col_name, phenotypes_keys, base_keys):
#     phenotype_data_per_base = {}
#     for gen in base_keys:
#         phenotype_data_per_base[gen] = {}
#         for key in phenotypes_keys:
#             phenotype_data_per_base[gen][key] = data[data[data_col_name] == gen][key].tolist()
    
#     return phenotype_data_per_base

# def do_anova(data):
#     df = pd.DataFrame.from_dict(data, orient='index')
#     output = {}
    
#     for key in df:
#         fvalue, pvalue = stats.f_oneway(*df[key])
#         output[key] = {}
#         output[key]["fvalue"] = round(fvalue, 4)
#         output[key]["pvalue"] = round(pvalue, 4)
        
#     return output

# def detect_pvalue(data):
#     for key in data:
#         if data[key]['pvalue'] <= 0.05:
#             return True, key
    
#     return False, ''
        
# def get_anova_file(data, phenotypes_df, START_SEQ_ROW = 15):
#     index = 0
#     phenotypes_keys = phenotypes_df.keys()[2:-1]
    
#     overall_anova = []
#     for column in data.columns[START_SEQ_ROW:]:
#         results = {}
        
#         base_keys = data[column].unique().tolist()
#         if(len(base_keys) == 1):
#             continue
        
#         phenotype_data_per_base = get_phenotype_data_per_base(data, column, phenotypes_keys, base_keys)
#         output = do_anova(phenotype_data_per_base)
#         value, key_name = detect_pvalue(output)
        
#         results['index'] = index
#         results['col'] = column
#         results['significance'] = {
#             'value': value,
#             'key_name': key_name
#         }
#         results['anova_data'] = output
        
#         index += 1
#         overall_anova.append(results)
    
#     return overall_anova


# def clean_data_based_on_anova(anova_results, data, phenotypes_df):
#     align = []
#     indexes_to_remove = []
    
#     for res in anova_results:
#         if res["significance"]["value"] == True:
#             indexes_to_remove.append(res["index"])
            
#     for seq in data:
#         if seq.id in phenotypes_df["Standard"].tolist():
#             sequence_to_clean = list(str(seq.seq))
        
#             for i in indexes_to_remove:
#                 sequence_to_clean[i] = ''
                
#             cleaned_seq = "".join(sequence_to_clean)
#             align.append(SeqRecord(Seq(cleaned_seq), id=seq.id))
            
#     return MultipleSeqAlignment(align)


# def write_phylip_file(data, path, filename):
#     if data == []:
#         print("No data.")
#         return False
    
#     with open(f"{path}\\{filename}.phylip", "w") as handle:
#         AlignIO.write(data, handle, "phylip-sequential")


# def do_anova_task(path, phenotypes_df):
#     ANOVA_RESULTS = "C:\\Users\\Hector\\Desktop\\code\\sc_regmod\\dataset\\anova_results"

#     #print(f"Thread starting with ind: {counter}, file: {path}")
    
#     start_time = time.time()
    
#     data = read_phylip_file(path)
    
#     anova_data = prepare_dataframe(data, phenotypes_df, "all")
    
#     overall_data = get_anova_file(anova_data, phenotypes_df)
    
#     new_align = clean_data_based_on_anova(overall_data, data, phenotypes_df)
    
#     write_phylip_file(new_align, ANOVA_RESULTS, path.split("\\")[-1])
    
#     print("Create the file {} took: {:.2f} sec".format(path.split("\\")[-1], (time.time() - start_time)))
