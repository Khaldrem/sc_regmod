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


def get_anova_filepaths(base_path="", type_anova="at_least_one", p_value=0.05, chromosome="all"):
    final_path= base_path + "/" + \
                "anova_" + type_anova + "_phenotype" + "/" + \
                "p_value_" + str(p_value).replace(".", "_") + "/" + \
                chromosome

    if os.path.exists(final_path):
        return final_path
    
    return ""


def get_models_path(base_path = ""):
    check_directory(base_path)

    dirs = []
    for it in os.scandir(base_path):
        if it.is_dir():
            dirs.append(it)

    if check_working_os():
        check_directory(f"{base_path}/{len(dirs)}")
        return f"{base_path}/{len(dirs)}"
    else:
        check_directory(f"{base_path}/{len(dirs)}")
        return f"{base_path}\\{len(dirs)}"
