import os
from sys import platform 

def get_all_filenames(filepaths = []):
    all_filenames = []
    for fp in filepaths:
        all_filenames.append(get_filename(fp))
    return all_filenames

def get_filename(filepath=""):
    if check_working_os():
        return filepath.split("/")[-1].split(".")[0]
    else:
        return filepath.split("\\")[-1].split(".")[0]


def check_directory(dir_path=""):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def check_working_os():
    """
        Returns true if working on linux
    """
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        return True
    
    return False