from os import sep
import pandas as pd

def read_phenotypes_file(filepath=""):
    return pd.read_csv(filepath, sep=";")