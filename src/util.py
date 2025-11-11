import importlib
import csv
import os


def read_all_files(folder_path):
    result = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            result.append(file_path)
    return result


def read_first_column(csv_path:str):
    if not csv_path.endswith(".csv"): return [] # If the input is not csv, there's no "first column" so return nothing, also trash in trash out.
    first_col = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # skip empty lines
                first_col.append(row[0])
    return first_col


def load_class(path: str):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls

