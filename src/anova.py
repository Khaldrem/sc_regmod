from multiprocessing import Pool
from functools import partial
from datetime import datetime
import time
import pandas as pd
import numpy as np
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from scipy.stats import f_oneway


from src.io import check_directory, check_working_os, get_filepaths, read_phenotypes_file, read_phylip_file, write_phylip_file, write_pandas_csv
from src.utils import get_filename, get_all_filenames
from src.filters import eliminate_columns_based_on_list
from src.indexes import insert_length, insert_col_positions_data

VALID_PHENOTYPES = ["SM300-Efficiency", "SM300-Rate", "SM300-Lag", "SM300-AUC",
                    "SM60-Efficiency",  "SM60-Rate",  "SM60-Lag",  "SM60-AUC",
                    "Ratio-Efficiency", "Ratio-Rate", "Ratio-Lag", "Ratio-AUC"]
VALID_CHROMOSOMES = ["haploide-euploide", "diploides-euploides", "---", "all"]
log_results = []


def do_folder_validations(with_phenotypes, phenotype, p_value, chromosome, ANOVA_DATASET_PATH):
    if with_phenotypes == "particular": 
        if phenotype not in VALID_PHENOTYPES:
            print(f"phenotype: {phenotype} is not valid.")
            return ""

    if chromosome not in VALID_CHROMOSOMES:
        print(f"chromosome: {chromosome} is not valid.")
        return ""
    
    #Check si existe el directorio principal
    check_directory(ANOVA_DATASET_PATH)

    #Chequeo si el directorio que separa los anovas a realizar existe
    WITH_PHENOTYPES_DATASET_PATH = ""
    if with_phenotypes == "at_least_one":
        if check_working_os():
            WITH_PHENOTYPES_DATASET_PATH = ANOVA_DATASET_PATH + "/" + "anova_at_least_one_phenotype"
        else:
            WITH_PHENOTYPES_DATASET_PATH = ANOVA_DATASET_PATH + "\\" + "anova_at_least_one_phenotype"


    if with_phenotypes == "particular":
        if check_working_os():
            WITH_PHENOTYPES_DATASET_PATH = ANOVA_DATASET_PATH + "/" + "anova_particular_phenotype"
        else:
            WITH_PHENOTYPES_DATASET_PATH = ANOVA_DATASET_PATH + "\\" + "anova_particular_phenotype"

    check_directory(WITH_PHENOTYPES_DATASET_PATH)

    #Chequeo que el directorio para ese p_value exista
    PVALUE_DATASET_PATH = ""
    if check_working_os():
        PVALUE_DATASET_PATH = WITH_PHENOTYPES_DATASET_PATH + "/" + "p_value_" + str(p_value).replace(".", "_")
    else:
        PVALUE_DATASET_PATH = WITH_PHENOTYPES_DATASET_PATH + "\\" + "p_value_" + str(p_value).replace(".", "_")

    check_directory(PVALUE_DATASET_PATH)

    #chequeo que el directorio para ese chromosoma exista
    CHROMOSOME_DATASET_PATH = ""
    if check_working_os():
        CHROMOSOME_DATASET_PATH = PVALUE_DATASET_PATH + "/" + chromosome
    else:
        CHROMOSOME_DATASET_PATH = PVALUE_DATASET_PATH + "\\" + chromosome

    check_directory(CHROMOSOME_DATASET_PATH)

    #A;ado a la ruta la carpeta del fenotipo si estoy en el caso de 1 solo en particular
    FINAL_PATH = ""
    if with_phenotypes == "particular":
        if check_working_os():
            FINAL_PATH = CHROMOSOME_DATASET_PATH + "/" + phenotype
        else:
            FINAL_PATH = CHROMOSOME_DATASET_PATH + "\\" + phenotype

    if with_phenotypes == "at_least_one":
        if check_working_os():
            FINAL_PATH = CHROMOSOME_DATASET_PATH
        else:
            FINAL_PATH = CHROMOSOME_DATASET_PATH

    check_directory(FINAL_PATH)

    #Creamos el directorio para los archivos csv con los resultados del anova
    CSV_FINAL_PATH = ""
    if check_working_os():
        CSV_FINAL_PATH = FINAL_PATH + "/" + "csv"
    else:
        CSV_FINAL_PATH = FINAL_PATH + "\\" + "csv"

    check_directory(CSV_FINAL_PATH)

    return FINAL_PATH, CSV_FINAL_PATH


def filter_filepaths(filepaths = [], DATASET_PATH = ""):
    dataset_filepaths = get_filepaths(DATASET_PATH)
    dataset_filenames = get_all_filenames(dataset_filepaths)

    filtered_list = []
    for fp in filepaths:
        filename = get_filename(fp)
        if filename not in dataset_filenames:
            filtered_list.append(fp) 

    return filtered_list


def order_phenotypes_by_files_id(example_filepath, phenotypes_df):
    data = read_phylip_file(example_filepath)
    id_rows = []
    for row in data:
        id_rows.append(row.id)

    df_mapping = pd.DataFrame({"ids": id_rows})
    sort_mapping = df_mapping.reset_index().set_index('ids')

    phenotypes_df["Standard_num"] = phenotypes_df["Standard"].map(sort_mapping["index"])
    phenotypes_df = phenotypes_df.sort_values("Standard_num")

    return phenotypes_df


def order_phenotypes_by_id_rows(phenotypes_df, id_rows):
    #Remove rows that are not in the index file
    phenotypes_df = phenotypes_df[phenotypes_df['Standard'].isin(id_rows)]

    # #Order by id_rows
    phenotypes_df = phenotypes_df.sort_values('Standard', key=lambda x: x.map({v:k for k, v in enumerate(id_rows)}))
    phenotypes_df = phenotypes_df.reset_index()

    #drop columnas innecesarias
    phenotypes_df = phenotypes_df.drop(['index', 'Standard', 'Haploide-Diploide', 'Ecological info'], axis=1)

    return phenotypes_df


def filtered_data(data, phenotypes_df):
    filtered_data = []
    ids = phenotypes_df["Standard"].tolist()
    for row in data:
        if row.id in ids:
            filtered_data.append(SeqRecord(row.seq, id=row.id))
    
    return MultipleSeqAlignment(filtered_data)


def detect_pvalue(threshold, pvalue):
    if pvalue <= threshold:
        return True
    return False
 

def do_anova_per_column(col, bases, df, phenotypes_to_do, anova_res_df, 
            cols_to_eliminate, cols_not_eliminated, p_value, type_anova,
            filename):
    #Las agrego los datos al df a trabajar
    df["bases"] = bases

    #Obtengo las bases unicas para esa columna
    unique_bases = set(bases)

    pvalue_detections = []
    for pt in phenotypes_to_do:
        to_pass = []
        for key in unique_bases:
            temp = df.loc[df["bases"] == key][pt].tolist()
            to_pass.append(temp)
        
        if len(to_pass) >= 2:
            anova_result = f_oneway(*to_pass)

            #Agregamos la info de la operacion al df de resultados del anova
            anova_res_df.loc[anova_res_df["col"] == col, pt + "_statistic"] = anova_result.statistic
            anova_res_df.loc[anova_res_df["col"] == col, pt + "_pvalue"] = anova_result.pvalue

            #Decidimos si dejamos o no la columna
            #Se comporta diferente para 'at_least_one' o para 'particular':
            if type_anova == "particular":
                if not detect_pvalue(p_value, anova_result.pvalue):
                    #Columna no guardada
                    cols_to_eliminate.append(col)
                else:
                    #Columna guardada
                    cols_not_eliminated.append(col)
            
            if type_anova == "at_least_one":
                pvalue_detections.append(detect_pvalue(p_value, anova_result.pvalue))
        # else:
        #     print(f"File: {filename} no presenta variacion en la columna {col}")
    
    if type_anova == "at_least_one":
        if True in pvalue_detections:
            cols_not_eliminated.append(col)
        else:
            cols_to_eliminate.append(col)


def do_anova(type_anova, phenotype, p_value, chromosome, phenotypes_df, 
        DATASET_PATH, CSV_DATASET_PATH, INDEX_PATH, filepath):
    filename = get_filename(filepath)
    data = read_phylip_file(filepath)

    phenotypes_df = order_phenotypes_by_files_id(filepath, phenotypes_df)

    # print(f"File: {filename} - alignment length: {data.get_alignment_length()}")

    #Filtro los datos si cambia la seleccion del cromosoma
    #Debido a que esto reduce la cantidad de filas
    if chromosome != "all":
        data = filtered_data(data, phenotypes_df)

    #Df que guarda los resultados de los anova por columna
    anova_res_df = pd.DataFrame({"col": range(data.get_alignment_length())})

    if type_anova == "at_least_one":
        #Agrego al anova_res_df las columnas de los fenotipos
        #tanto en el statistic y el pvalue
        for pt in VALID_PHENOTYPES:
            anova_res_df[pt + "_statistic"] = np.zeros(data.get_alignment_length())
            anova_res_df[pt + "_pvalue"] = np.zeros(data.get_alignment_length())

    if type_anova == "particular":
        #Agrego al anova_res_df la columna del fenotipo
        #tanto en el statistic y el pvalue
        anova_res_df[phenotype + "_statistic"] = np.zeros(data.get_alignment_length())
        anova_res_df[phenotype + "_pvalue"] = np.zeros(data.get_alignment_length())
    
    #Listado que almacena los indices de las columnas que se filtran
    cols_to_eliminate = []
    cols_not_eliminated = []

    #Creo un dataframe con los datos 
    df = phenotypes_df.copy()
    for col in range(data.get_alignment_length()):
        #Extraigo las bases de esa columna
        bases = list(data[:, col])

        if type_anova == "at_least_one":
            do_anova_per_column(col, bases, df, VALID_PHENOTYPES, 
                    anova_res_df, cols_to_eliminate, cols_not_eliminated, 
                    p_value, type_anova, filename)
        
        if type_anova == "particular":
            do_anova_per_column(col, bases, df, [phenotype], 
                    anova_res_df, cols_to_eliminate, cols_not_eliminated, 
                    p_value, type_anova, filename)

    #Consultados si tenemos elementos a eliminar, si es asi
    #Construimos los nuevos datos y los guardamos
    file_length = 0
    if cols_to_eliminate != []:
        new_data = eliminate_columns_based_on_list(data, cols_to_eliminate)
        file_length = new_data.get_alignment_length()

        if file_length != 0:
            #Guardamos los nuevos datos
            write_phylip_file(new_data, DATASET_PATH, filename)

            #Guardamos el csv con los datos obtenidos de los anova realizados
            write_pandas_csv(anova_res_df, CSV_DATASET_PATH, filename)

            #Actualizamos el indice del archivo, para guardar las columnas y el largo del nuevo archivo
            insert_col_positions_data(cols_not_eliminated, "anova", INDEX_PATH, filename)
            insert_length(file_length, "anova", INDEX_PATH, filename)
    #     else:
    #         print(f"        + File: {filename} elimina todas sus columnas. Por file_length: {file_length}")
            
    # else:
    #     print(f"        + File: {filename} elimina todas sus columnas. Por cols_to_eliminate: {len(cols_to_eliminate)}")



def log_do_anova(t):
    log_results.append(t)


def do_multiprocess_anova(type_anova="particular", phenotype="", p_value=0.05, 
                        chromosome="", pool_workers=1, 
                        ANOVA_DATASET_PATH="", CLEAN_DATASET_PATH="", 
                        PHENOTYPES_PATH="", INDEX_PATH=""):
    """
        type_anova: Posibles valores "particular" o "at_least_one", se refiere a si se hara un anova
                    considerando un solo fenotipo en particular (particular) o al que pase al menos 1 (at_least_one)
        phenotype: Escoge el fenotipo a realizar, si se realiza sobre todos los fenotipos, dejar en blanco
        p_value: Valor discriminante en el resultado del anova
        chromosome: Filtra por tipo de cromosomas las columnas, valores posibles: ["haploide-euploide", "diploides-euploides", "---", "all"]
        pool_workers:  
    """

    DATASET_PATH, CSV_DATASET_PATH = do_folder_validations(type_anova, phenotype, p_value, chromosome, ANOVA_DATASET_PATH)

    filepaths = get_filepaths(CLEAN_DATASET_PATH)
    filtered_filepaths = filter_filepaths(filepaths, DATASET_PATH)

    #Cargo el dataframe del archivo de fenotipos
    phenotypes_df = read_phenotypes_file(PHENOTYPES_PATH)
    
    #Filtro los datos dependiendo del cromosoma que ocupare
    if chromosome != "all":
        phenotypes_df = phenotypes_df.loc[phenotypes_df["Haploide-Diploide"] == chromosome]

    
    #Filtro el para los fenotipos
    if type_anova == "particular":
        phenotypes_df = phenotypes_df[["Standard", phenotype]]
    
    print(f"    + Creating ANOVA files ... ")
    # print(f"===== Start: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}  ======")
    start = time.time()
          
    p = Pool(pool_workers)
    # p.map_async(partial(do_anova, type_anova, phenotype, p_value, chromosome, phenotypes_df, 
    #         DATASET_PATH, CSV_DATASET_PATH, INDEX_PATH), filtered_filepaths)

    # p.map(partial(do_anova, type_anova, phenotype, p_value, chromosome, 
    #         phenotypes_df, DATASET_PATH, CSV_DATASET_PATH, INDEX_PATH), filtered_filepaths)

    p.imap(partial(do_anova, type_anova, phenotype, p_value, chromosome, phenotypes_df, 
                        DATASET_PATH, CSV_DATASET_PATH, INDEX_PATH), filtered_filepaths)

    # for fp in filtered_filepaths:
    #     p.apply_async(do_anova, args=(type_anova, phenotype, p_value, chromosome, phenotypes_df, 
    #                             DATASET_PATH, CSV_DATASET_PATH, INDEX_PATH, fp, ), callback=log_do_anova)

    p.close()
    p.join()


    end = time.time()
    print(f"    + Files created in: {round((end-start)/60, 2)} min.")
    print()
