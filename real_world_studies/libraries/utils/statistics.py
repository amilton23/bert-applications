"""
Biblioteca para análises estatísticas mais comuns em cientistas de dados - tanto output jupyter notebook quanto outras aplicações 
"""

import argparse
import logging
import json
import os
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from nltk import sent_tokenize
from unidecode import unidecode
import re 

import tomli
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import matplotlib.pyplot as plt

# Avaliação do modelo
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    multilabel_confusion_matrix,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)

def print_classification_metrics(
    y_true, 
    y_pred, 
    is_multiclass=False
):
    """
    Calcula e imprime as métricas de avaliação (Accuracy, Precision, Recall, F1 Score)
    com suporte para classificação multiclasses e binária.

    Parâmetros:
    - data_poc_enriched_final: DataFrame contendo as colunas 'y' e 'VL_FLAG' (verdadeiro e predito).
    - is_multiclass: booleano, indica se o problema é multiclasses. 
      Se True, as métricas serão calculadas com média ponderada.
    """

    # Calcula e imprime métricas para classificação multiclasses
    if is_multiclass:
        print('Accuracy:', accuracy_score(y_true, y_pred))
        print('Precision:', precision_score(y_true, y_pred, average='weighted'))
        print('Recall:', recall_score(y_true, y_pred, average='weighted'))
        print('F1 Score:', f1_score(y_true, y_pred, average='weighted'))
    else:
        # Calcula e imprime métricas para classificação binária
        print('Accuracy:', accuracy_score(y_true, y_pred))
        print('Precision:', precision_score(y_true, y_pred))
        print('Recall:', recall_score(y_true, y_pred))
        print('F1 Score:', f1_score(y_true, y_pred))
    
    return None

def print_auc_and_plot_precision_recall(data_poc_enriched_final, is_multiclass=False):
    """
    Calcula o AUC (Área sob a Curva ROC) e, no caso de classificação binária, plota a curva Precision-Recall.
    Suporta tanto problemas de classificação multiclasses quanto binária.

    Parâmetros:
    - data_poc_enriched_final: DataFrame contendo as colunas 'y' (verdadeiros) e 'VL_SCORE' (probabilidade predita).
    - is_multiclass: booleano, indica se o problema é multiclasses. 
      Se True, será usado o método One-vs-Rest (OvR) para AUC multiclasses.
    """
    try:
        if is_multiclass:
            # AUC para problemas multiclasses usando a estratégia One-vs-Rest (OvR)
            print('AUC:', roc_auc_score(data_poc_enriched_final['y'], 
                                        data_poc_enriched_final['VL_SCORE'], 
                                        multi_class='ovr'))
        else:
            # AUC para problemas binários
            print('AUC:', roc_auc_score(data_poc_enriched_final['y'], 
                                        data_poc_enriched_final['VL_SCORE']))

            # Plotando a curva Precision-Recall para problemas binários
            precision, recall, _ = precision_recall_curve(data_poc_enriched_final['y'], 
                                                          data_poc_enriched_final['VL_SCORE'])

            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')

            # Salvando e exibindo a figura
            plt.savefig('training_precision_recall_curve.png')
            plt.show()

    except Exception as e:
        print("Ocorreu um erro ao calcular o AUC ou ao plotar a curva Precision-Recall.")
        print("Detalhes do erro:", e)
        
    return None

def df_downsample(df, col:str, rand_state: bool = True):
    """
    Description:
        Função que realiza balanceamento para a mínima frequência (downsample) a partir da coluna de um dataframe. 
    Args:
        df  - (pandas.DataFrame) pandas DataFrame com objetivo de realizar o downsample.
        col - (str) coluna objetivo do downsample.
        rand_state - (True/False) variável que identifica se a amostra é aleatorizada totalmente ou parcialmente (indicando um random_state fixo no sample). Recebe binário (True ou False)
    Return:
        df_strat - pandas DataFrame estratificado resultante do downsample.
    """
    col = str(col)
    val_counts = df[col].value_counts()
    freq_min = val_counts.min()
    #qtd_pos = len(df[df[col] == 1])
    
    if rand_state:
        df_strat = df.groupby(col, group_keys=False).apply(lambda x: x.sample(freq_min, random_state=2))
    else:
        df_strat = df.groupby(col, group_keys=False).apply(lambda x: x.sample(freq_min))
        
    return df_strat

def confidence_sample_size(N, z = 1.96, p = 0.5, Cp = 0.05):
    """
    Função de encontrar a melhor representatividade de tamanho amostral em função da população.
    z table pos:
        2.575 - 99% confiança.
        1.96  - 95% conf.
        1.75  - 92% conf.
    Cp - margem de erro.
    z - z-score ideal (vide z-table positiva)
    p - proporção da variável estudada (ex.: 20% dos casos da população são câncer de cólo do útero, p = 0.2. OBS.: Caso não saiba p, considere 0.5 - mais conservador.)
    """
    sample_size = ((z)**2*(p*(1-p))*N) / ((z)**2*(p*(1-p)) + (N-1)*(Cp)**2)
    return sample_size

def interest_proportion(
    df: pd.DataFrame or pd.Series, 
    col: str, 
    interest_value: str or int
    ):
    """
    Description:
        Gera a proporção da nossa resposta de interesse para certos cálculos estatísticos.
    Args:
        df - DataFrame ou vetor onde a amostra com as respostas está armazenada.
        col - Coluna de interesse para o estudo.
        interest_value - Valor de interesse do estudo como está declarado na coluna. Ex.: Coluna "y", minha resposta de interesse é 1.
    Example:
        Desejo saber o tamanho amostral de um certo exame porém não tenho a população, porém tenho a proporção do grupo controle e do grupo teste, necessita-se 
        da diferença da proporção de resposta de interesse da cada grupo.
    """
    p = df[col][df[col] == interest_value].count() / len(df[col])
    return p

def power_analysis(p1 = 0, p2 = 1, proportion_use=False, effect_size = 0.5, alpha=0.01, power=0.80):
    """
    Description:
        Teste T para análise de poder para o tamanho amostral ideal em distribuições não normais ou menores que 30.
        
        Obs.: Para Anatomopatológicos, effect_size = 0.110333 por ser a média dos cânceres estudados segundo estudo do INCA de 2023.
        "O tumor maligno mais incidente no Brasil é o de pele não melanoma (31,3% do total de casos), seguido pelos de mama feminina (10,5%), 
        próstata (10,2%), cólon e reto (6,5%), pulmão (4,6%) e estômago (3,1%)."
        
        Fonte: 
        https://bvsms.saude.gov.br/inca-lanca-a-estimativa-2023-incidencia-de-cancer-no-brasil/#:~:text=O%20tumor%20maligno%20mais%20incidente,est%C3%B4mago%20(3%2C1%25).
    Args:
        
    Example:
    """
    # Calcular a proporção combinada
    p = (p1 + p2) / 2
    
    if (abs(p1 - p2) == 0) or (proportion_use==False):
        nobs = sm.stats.NormalIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1, alternative='two-sided')       
    else:
        # Calcular o effect size usando a fórmula para proporções
        effect_size = abs(p1 - p2) / np.sqrt(p * (1 - p))
        nobs = sm.stats.NormalIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1, alternative='two-sided')
    return round(nobs)