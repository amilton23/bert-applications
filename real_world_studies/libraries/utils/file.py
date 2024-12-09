"""
Package de uso de dados e arquivos. 

"""

import datetime
import hashlib
import math
from datetime import datetime, timezone
from typing import List, TypeVar

import numpy as np
import pandas as pd
import re 
import os 
import json

T = TypeVar("T")

def load_local_key(keys_path, key_name):
    """
    Caso não esteja utilizando credenciais em repositório.
    """
    if os.path.exists(keys_path):
        with open(keys_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            if key_name not in data or data[key_name] is None or data[key_name] == "":
                print(f"Chave {key_name} não encontrada ou vazia.")
                return None
            return data[key_name]
    return None

def create_dir_returning_path(name:str):
    """
    Cria pasta de nome passado como string. Função utilitária de model_evaluation_cm().
    
    Args:
        name (str): nome da pasta em formato de string
        
    Return:
        path (str): caminho da pasta em string para utilização desta
    """
    path_dir = os.getcwd() + f'/{name}/'

    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    
    return path_dir

def switch_label(dict_mapping, series):
    """
    Altera label string para resposta quantitativa a partir de dict, saída é uma pandas Series.
    """
    # Inverter o dicionário para ter as chaves como valores e vice-versa
    #inverted_dict = {v: k for k, v in dict_mapping.items()}
    
    # Mapear os rótulos usando o dicionário invertido
    result_series = series.map(dict_mapping)
    
    # Substituir os valores não mapeados por NaN
    result_series = result_series.fillna(series)    
    
    return result_series

def get_current_datetime():
    return datetime.now(timezone.utc)

if __name__ == "__main__":
    pass
