"""
Biblioteca para visualização de dados - tanto output jupyter notebook quanto outras aplicações 
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
import ocifs
import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
plt.style.use("seaborn-white")

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

from .statistics import (
    confidence_sample_size, 
    power_analysis,
    interest_proportion,
)

from .file import (
    create_dir_returning_path, 
)

def model_evaluation_cm(
    y_true,
    y_predicted, 
    disp_labels=None, 
    label_mapping=None, 
    filename='cm_00.png', 
    folder_name='', 
    suptl='CONFUSION MATRIX', 
    xlabel='', 
    ylabel='',
    cm_legend = True):
    """
    Função para criar e salvar um gráfico de matriz de confusão com rótulos transcritos e indicação de 
    VP, VN, FP e FN dentro dos quadrantes.

    Args:
        y_true: Série ou lista com os valores reais.
        y_predicted: Série ou lista com os valores previstos pelo modelo.
        disp_labels: Lista de rótulos opcionais para os eixos.
        label_mapping: Dicionário para mapear rótulos para descrições (ex: {"0": "ÓRGÃO NÃO IDENTIFICADO", "1": "ÓRGÃO IDENTIFICADO"}).
        filename: Nome do arquivo de saída para salvar a matriz de confusão.
        folder_name: Nome da pasta para salvar o arquivo.
        suptl: Título principal do gráfico.
        xlabel: Rótulo personalizado para o eixo x.
        ylabel: Rótulo personalizado para o eixo y.

    Retorna:
        None
    """
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))

    # Definindo os rótulos como únicos valores de y_true se não forem fornecidos
    if disp_labels is None:
        disp_labels = sorted(list(pd.unique(y_true)))

    # Se um mapeamento de rótulos foi fornecido, aplicamos as descrições
    if label_mapping is not None:
        disp_labels = [label_mapping.get(str(label), str(label)) for label in disp_labels]

    # Calculando a matriz de confusão
    cm = confusion_matrix(y_true=y_true, y_pred=y_predicted, labels=sorted(pd.unique(y_true)))

    # Gerando a matriz de confusão com uma escala de cores e fixando os rótulos
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    disp.plot(cmap='Blues', ax=axs, values_format='d')  # 'd' para valores inteiros fixos

    # Adicionando rótulos de VP, VN, FP, FN no centro de cada quadrante com ajuste de cor da fonte
    max_value = cm.max()  # Para normalizar as cores e definir a legibilidade

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Definindo a cor da fonte com base na intensidade
            color = "white" if cm[i, j] > max_value / 2 else "black"

            if cm.shape == (2, 2):  # Caso binário
                if i == j:
                    label = "VP" if i == 1 else "VN"  # Diagonal principal
                else:
                    label = "FP" if i < j else "FN"  # Fora da diagonal
            else:
                label = ""
            
            # Adicionando o texto e escolhendo a cor da fonte
            axs.text(j, i, f"\n\n({label})", ha="center", va="center", color=color, fontsize=10)

    # Ajustando o título, eixos x e y
    axs.set_title(suptl, fontsize=14)
    axs.set_xlabel(xlabel if xlabel else 'Predição da IA', fontsize=12)
    axs.set_ylabel(ylabel if ylabel else 'Predição Real', fontsize=12)

    # Configurando os limites dos eixos para manter os quadrantes fixos
    axs.set_xlim(-0.5, len(disp_labels) - 0.5)
    axs.set_ylim(len(disp_labels) - 0.5, -0.5)  # Invertido para alinhar com o display padrão

    # Criando a pasta de saída para salvar o gráfico
    OUTPUT_DIR = create_dir_returning_path(folder_name if folder_name else 'outputs')
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=600, bbox_inches='tight')

    # Exibindo o relatório de classificação
    print(classification_report(y_true=y_true, y_pred=y_predicted))
    plt.show()
    plt.close(fig)  # Fechar o gráfico após salvar para evitar sobreposições em gráficos futuros
    return None

def plot_roc_curve(y_true, y_prob, is_multiclass=False):
    """
    Plota a curva ROC para classificação binária e exibe uma mensagem informativa para classificação multiclasses.
    Salva o gráfico da curva ROC em um arquivo no caso de classificação binária.

    Parâmetros:
    - df: DataFrame contendo as colunas 'y_true' (verdadeiros) e 'y_prob' (probabilidade predita).
    - is_multiclass: booleano, indica se o problema é multiclasses. 
      Se True, exibe uma mensagem informando que o gráfico ROC multiclasses não foi implementado.
    """
    try:
        if is_multiclass:
            # Exibe uma mensagem caso o gráfico ROC multiclasses não esteja implementado
            print("Gráfico de curva ROC para modelo multiclass ainda não implementado.")
        else:
            # Calcula e plota a curva ROC para classificação binária
            fpr, tpr, thresholds = roc_curve(y_true, 
                                             y_prob)

            plt.plot(fpr, tpr, label="ROC Curve")
            plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()

            # Salva e exibe a figura da curva ROC
            plt.savefig('training_roc_curve.png')
            plt.show()

    except Exception as e:
        print("Ocorreu um erro ao calcular ou plotar a curva ROC.")
        print("Detalhes do erro:", e)
        
    return None

def barplot_series(
    series: pd.Series, 
    is_filtered:bool = False, 
    rank_filter:str = '',
    file_name:str = None
    ):
    """
    Cria um gráfico de barras a partir de um pandas Series.
    
    Args:
        series: pd.Series - 
        is_filtered : bool (default False) -  
        rank_filter:str (default '') - 
        file_name:str (default None) -
        
    Return:
        None
    """
    # Aplicar o filtro se is_filtered=True e col_filter for válido
    if is_filtered:      
        # Verifica se rank_filter tem um "+" (maiores) ou "-" (menores)
        if rank_filter.endswith("+"):
            n = int(rank_filter[:-1])
            series = series.nlargest(n)
        elif rank_filter.endswith("-"):
            n = int(rank_filter[:-1])
            series = series.nsmallest(n)
        else:
            raise ValueError("O rank_filter deve terminar com '+' para os maiores ou '-' para os menores.")    
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 12), sharex=True)
    bar_width = 0.5
    index = np.arange(len(series))  # Usamos o series filtrado

    # Gráfico de barras para Contagem
    # Gráfico de Barras Empilhadas para y_1 e y_0
    ax.bar(index - bar_width/2, series, bar_width, color='skyblue', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Frequência dos termos')
    ax.set_xticks(index)
    ax.set_xticklabels(series.index, rotation=45, ha="right")

    max_value = round(int(series.max()))

    # Função para arredondar para múltiplos de 10
    def arredondar_para_10(x):
        return np.round(x / 10) * 10

    # Gerar valores de 5% em 5% até o máximo
    yticks = np.arange(0, max_value + 1, max_value / 20)

    # Arredondar os valores dos ticks para múltiplos de 10
    yticks_arredondados = arredondar_para_10(yticks)

    ax.set_yticks(yticks_arredondados)   
    plt.tight_layout()
    
    
    file_path = os.getcwd()
    
    plt.savefig(os.path.join(file_path, f"{file_name}_{pd.to_datetime('today').strftime('%Y_%m_%d')}.png"), dpi=600, bbox_inches='tight')
    print(f"Gráfico salvo no path: {file_path}{file_name}_{pd.to_datetime('today').strftime('%Y_%m_%d')}.png")    
    
    plt.show()
    return None

def bar_of_pie(
    df, 
    col, 
    interest_col, 
    threshold, 
    file_path,
    settitle = '',
    file_name = None):
    """
    Gera um gráfico de barra ao lado de um gráfico de pizza (bar of pie).
    O gráfico mostra a proporção da parcela de interesse com base no threshold dado.

    Parâmetros:
    - df: DataFrame contendo os dados.
    - col: Coluna categórica para o gráfico de barras.
    - interest_col: Coluna numérica para categorizar o interesse (ex: acurácia).
    - threshold: Limite de valor para a categorização da parcela de interesse.
    - settitle: Título do gráfico.
    - file_path: Caminho para salvar o gráfico gerado.
    - file_name: nome do arquivo.
    """
    
    # Categorizar a coluna de interesse
    df['Acima_Limite'] = df[interest_col] >= threshold
    total = len(df)
    
    # Contar os valores acima e abaixo do limite
    acima_limite = df['Acima_Limite'].sum()
    abaixo_limite = total - acima_limite

    # Calculando proporções
    overall_ratios = [acima_limite / total, abaixo_limite / total]
    labels = [f'Acima de {threshold}', f'Abaixo de {threshold}']

    # Filtrar para remover valores com 0%
    overall_ratios, labels = zip(*[(ratio, label) for ratio, label in zip(overall_ratios, labels) if ratio > 0])

    # Verificar quantos segmentos sobraram após o filtro
    if len(overall_ratios) == 1:
        startangle = 90  # Apenas uma fatia, ângulo fixo
    elif len(overall_ratios) > 1:
        # Garantir que a fatia de interesse (abaixo do limite) fique à direita
        startangle = 90 - (overall_ratios[1] * 360 / 2) + 90  # Ajuste para garantir que a parcela de interesse fique à direita
    else:
        print("Erro: Não há fatias para plotar.")
        return None

    # Proporções da barra (para as categorias de 'col' apenas dos que estão abaixo do limite)
    df_below_threshold = df[df[interest_col] < threshold]
    bar_ratios = df_below_threshold[col].value_counts(normalize=True).sort_values(ascending=False)
    bar_labels = bar_ratios.index.tolist()
    bar_values = bar_ratios.values

    # Criar a figura e os eixos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Diminuir o espaço entre os gráficos de pizza e barras
    fig.subplots_adjust(wspace=0.0)  # Aproxime os gráficos

    # Parâmetros para o gráfico de pizza
    explode = [0] * len(overall_ratios)
    if len(overall_ratios) > 1:
        explode[1] = 0.1  # Explodir a fatia de interesse se houver mais de uma fatia

    wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=startangle,
                         labels=labels, explode=explode, colors=['#66b3ff', '#ff9999'])

    # Gráfico de barras (para os procedimentos abaixo do limite)
    bottom = 1
    width = .1  # Diminuir a largura do gráfico de barras
    
    for j, (height, label) in enumerate(reversed([*zip(bar_values, bar_labels)])):
        bottom -= height
        alpha_value = min(0.1 + 0.25 * j, 1)  # Garante que alpha não seja maior que 1
        bc = ax2.bar(0, height, width, bottom=bottom, color='C0', alpha=alpha_value)
        
        # Adiciona o rótulo (procedimento) ao lado da barra com um pequeno espaço
        ax2.text(width * 1.02, bottom + height / 2, label, ha='left', va='center', fontsize=10)

        # Adiciona a porcentagem dentro da barra
        ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')
    
    # Ajustando título do gráfico de barras
    ax2.set_title(f'{col}')
        
    # Remover a legenda
    ax2.axis('off')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

#     # Conectar o gráfico de pizza com o gráfico de barras com uma seta
#     theta = (wedges[1].theta1 + wedges[1].theta2) / 2  # Ponto médio da fatia de interesse
#     center, r = wedges[1].center, wedges[1].r
#     x = r * np.cos(np.pi / 180 * theta) + center[0]
#     y = r * np.sin(np.pi / 180 * theta) + center[1]
#     bar_height = sum(bar_values)

#     # Ajustar a posição da seta
#     arrow_start = (x, y)
#     arrow_end = (-width / 2, bar_height)
    
#     # Desenhar a seta de conexão
#     arrow = FancyArrowPatch(arrow_start, arrow_end, 
#                             connectionstyle="arc3, rad=-1.5", 
#                             arrowstyle="->", 
#                             mutation_scale=15, 
#                             color='black')
#     ax2.add_patch(arrow)
    
    if settitle == '':
        fig.suptitle(f'Gráfico de pizza com barras condicional a partir de {col}')
    else:
        fig.suptitle(f'{settitle}')
        
    # Ajustar o layout e mostrar/salvar o gráfico
    plt.tight_layout()
    
    if file_name is None:
        plt.savefig(os.path.join(file_path, f"Piechart_with_bar_{col}_{pd.to_datetime('today').strftime('%Y_%m_%d')}.png"), dpi=600, bbox_inches='tight')
        print(f"Gráfico salvo no path: {file_path}Piechart_with_bar_{col}_{pd.to_datetime('today').strftime('%Y_%m_%d')}.png")
    else:
        plt.savefig(os.path.join(file_path, f"{file_name}_{pd.to_datetime('today').strftime('%Y_%m_%d')}.png"), dpi=600, bbox_inches='tight')
        print(f"Gráfico salvo no path: {file_path}{file_name}_{pd.to_datetime('today').strftime('%Y_%m_%d')}.png")
    plt.show()
    return None

def evalmetrics_multicategory(
    df: pd.DataFrame,
    col: str,
    file_path: str,
    y_true_col: str,
    y_pred_col: str,
    settitle: str,
    pop_included: bool = False,
    file_name=None,
    col_pop: str = '',
    df_pop_per_date: bool = False,
    fracao_periodo: float = 0.5,
    acc_limit: float = 0.95,
    effect_size: float = 0.110333,
    is_filtered: bool = False,
    col_filter: str = '',
    rank_filter: str = '',
    **kwargs
):
    """
    Description:
        Gera um DataFrame com testes de significância estatística Z e t para 99, 95 e 92%, bem como métricas de avaliação de modelo de Acurácia, Precisão e Revocação a partir de uma análise multicategórica da coluna.
        
    Args:
        df - DataFrame contendo a coluna respost
        col - Nome da coluna a ser analisada
        file_path - Caminho para salvar o arquivo
        pop_included - Indica se a população deve ser incluída
        col_pop - Nome da coluna de população
        df_pop_per_date - Indica se o DataFrame de população é por data
        fracao_periodo - Fração do período para ajustar o tamanho da população
        acc_limit - Limite de acurácia para a linha horizontal
        effect_size - Tamanho do efeito para análise de poder
        is_filtered - Indica se o DataFrame deve ser filtrado
        col_filter - Nome da coluna para filtragem
        rank_filter - Filtro para maiores ou menores valores
        **kwargs - Para adicionar um DataFrame de população, por exemplo.
        
    Return:
        df_resultados_total - DataFrame contendo cada resposta da análise de significância 
    """
    
    unique = df[col].unique()
    resultados = []
    contagem_testes = {'Tipo de Teste': [], 'Sucesso': [], 'Falha': []}

    for i in range(len(unique)):
        categoria_atual = unique[i]
        df_aux = df[df[col] == categoria_atual].reset_index(drop=True)
        y_true = df_aux[y_true_col]
        y_pred = df_aux[y_pred_col]
        sample_size = df_aux[col].value_counts().iloc[0]  # VALOR (TAMANHO)
        
        # Inicializar valores de teste Z com None para evitar erros de referência
        test_z_99 = None
        test_z_95 = None
        test_z_92 = None
        
        # Se população estiver incluída, calcular os valores do teste Z
        if pop_included:
            df_pop = list(kwargs.values())[0]
            try:
                if df_pop_per_date:
                    try:
                        pop_size = round(df_pop[(df_pop["MES"] == 8) & (df_pop[col_pop] == int(categoria_atual))]["TAMANHO"].iloc[0] * fracao_periodo)
                    except (IndexError, KeyError, ValueError):
                        pop_size = round(df_pop[(df_pop["MES"] == 8) & (df_pop[col_pop] == categoria_atual)]["TAMANHO"].iloc[0] * fracao_periodo)
                else:
                    try:
                        pop_size = round(df_pop[(df_pop[col_pop] == int(categoria_atual))]["TAMANHO"].iloc[0])
                    except (IndexError, KeyError, ValueError):
                        pop_size = round(df_pop[(df_pop[col_pop] == categoria_atual)]["TAMANHO"].iloc[0])

                # Testes Z
                test_z_99 = round(confidence_sample_size(pop_size, 2.575, 0.32))
                test_z_95 = round(confidence_sample_size(pop_size, 1.960, 0.32))
                test_z_92 = round(confidence_sample_size(pop_size, 1.750, 0.32))

            except (IndexError, KeyError, ValueError):
                pop_size = None
                test_z_99 = None
                test_z_95 = None
                test_z_92 = None

        # Calcular proporções e análise de poder
        p1 = interest_proportion(df_aux, y_true_col, 1)
        p2 = interest_proportion(df_aux, y_pred_col, 1)
        pwr_analysis_99 = power_analysis(p1, p2, proportion_use=True, alpha=0.01)
        pwr_analysis_95 = power_analysis(p1, p2, proportion_use=True, alpha=0.05)
        pwr_analysis_92 = power_analysis(p1, p2, proportion_use=True, alpha=0.08)

        # Métricas de desempenho
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        n_size = len(df_aux)

        # Significância para os testes z
        bool_testz_99 = test_z_99 is not None and test_z_99 <= n_size
        bool_testz_95 = test_z_95 is not None and test_z_95 <= n_size
        bool_testz_92 = test_z_92 is not None and test_z_92 <= n_size

        # Significância para a análise de poder
        bool_pwr_99 = pwr_analysis_99 <= n_size
        bool_pwr_95 = pwr_analysis_95 <= n_size
        bool_pwr_92 = pwr_analysis_92 <= n_size
        
        y_1 = len(df_aux[df_aux[y_pred_col] == 1])
        y_0 = len(df_aux[df_aux[y_pred_col] == 0])
        
        # Adiciona ao dataframe de resultados
        resultados.append({
            f'{col}': categoria_atual,
            'Tamanho amostral': n_size,
            'Tamanho população': pop_size if pop_included else None,
            'y_1': y_1,
            'y_0': y_0,
            'Significância teste z (99%)': bool_testz_99,
            'Significância teste z (95%)': bool_testz_95,
            'Significância teste z (92%)': bool_testz_92,
            'Significância análise de poder (99%)': bool_pwr_99,
            'Significância análise de poder (95%)': bool_pwr_95,
            'Significância análise de poder (92%)': bool_pwr_92,
            'Acurácia': acc,
            'Precisão': precision,
            'Recall': recall,
            'F1-Score': f1,
            'SIGNIFICANTE': int(any([bool_testz_99, bool_testz_95, bool_testz_92, bool_pwr_99, bool_pwr_95, bool_pwr_92]))  # 1 se qualquer teste deu True
        })

    # Converter resultados em DataFrame
    df_resultados = pd.DataFrame(resultados)
    df_resultados_total = df_resultados
    
    # Aplicar o filtro se is_filtered=True e col_filter for válido
    if is_filtered:
        if col_filter not in df_resultados.columns:
            raise ValueError(f"A coluna {col_filter} não existe no DataFrame.")
        
        if not pd.api.types.is_numeric_dtype(df_resultados[col_filter]):
            raise ValueError(f"A coluna {col_filter} precisa ser numérica (float ou int).")
        
        # Verifica se rank_filter tem um "+" (maiores) ou "-" (menores)
        if rank_filter.endswith("+"):
            n = int(rank_filter[:-1])
            df_resultados = df_resultados.nlargest(n, col_filter)
        elif rank_filter.endswith("-"):
            n = int(rank_filter[:-1])
            df_resultados = df_resultados.nsmallest(n, col_filter)
        else:
            raise ValueError("O rank_filter deve terminar com '+' para os maiores ou '-' para os menores.")

    # Gerar o gráfico com os dados filtrados
    fig, ax = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    bar_width = 0.5
    index = np.arange(len(df_resultados))  # Usamos o df_resultados filtrado
    
    # Primeiro gráfico para Acurácia
    ax[0].bar(index - bar_width/2, df_resultados['Acurácia'] * 100, bar_width, color='lightgreen', label='Acurácia', edgecolor='black', linewidth=0.5)
    ax[0].set_ylabel('Acurácia (%)')
    ax[0].set_ylim(0, 100)
    
    # Linha da acurácia
    ax[0].axhline(y=acc_limit * 100, color='red', linestyle='--', linewidth=1.5, label=f'Acurácia {acc_limit * 100}%')
    ax[0].text(len(df_resultados) - 0.5, (acc_limit - 0.005) * 100, f'Acurácia mínima área de negócio {acc_limit * 100}%', color='firebrick', fontsize=12, verticalalignment='top', horizontalalignment='right')

    yticks_02 = np.arange(0, 1.1, 0.05) * 100
    ax[0].set_yticks(yticks_02)

    ax[0].set_xticks(index)
    ax[0].set_xticklabels(df_resultados[col], rotation=45, ha="right")
    
    if is_filtered:
        ax[0].set_title(f'{settitle} (Filtrado: {col_filter}, {rank_filter})')
    else:
        ax[0].set_title(f'{settitle}')
    
    # Gráfico de barras para Contagem
    # Gráfico de Barras Empilhadas para y_1 e y_0
    ax[1].bar(index - bar_width/2, df_resultados['y_0'], bar_width, label='Negativos', color='skyblue', edgecolor='black', linewidth=0.5)
    ax[1].bar(index - bar_width/2, df_resultados['y_1'], bar_width, bottom=df_resultados['y_0'], label='Positivos', color='lightcoral', edgecolor='black', linewidth=0.5)

    ax[1].set_ylabel('Frequência do tamanho da amostra')
    ax[1].set_xticks(index)
    ax[1].set_xticklabels(df_resultados[col], rotation=45, ha="right")

    max_value = round(int(df_resultados['Tamanho amostral'].max()))

    # Função para arredondar para múltiplos de 10
    def arredondar_para_10(x):
        return np.round(x / 10) * 10

    # Gerar valores de 5% em 5% até o máximo
    yticks = np.arange(0, max_value + 1, max_value / 20)
    
    # Arredondar os valores dos ticks para múltiplos de 10
    yticks_arredondados = arredondar_para_10(yticks)

    ax[1].set_yticks(yticks_arredondados)    
    
    # Adicionar legenda
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.88), frameon=True, facecolor='white', edgecolor='black')

    # Ajustar layout e salvar o gráfico
    plt.tight_layout()
    
    if file_name is None:
        plt.savefig(os.path.join(file_path, f"barplot_count_{col}_{col_filter}{rank_filter}_{pd.to_datetime('today').strftime('%Y_%m_%d')}.png"), dpi=600, bbox_inches='tight')
        print(f"Gráfico salvo no path: {file_path}barplot_count_{col}_{col_filter}{rank_filter}_{pd.to_datetime('today').strftime('%Y_%m_%d')}.png")
    else:
        plt.savefig(os.path.join(file_path, f"{file_name}_{pd.to_datetime('today').strftime('%Y_%m_%d')}.png"), dpi=600, bbox_inches='tight')
        print(f"Gráfico salvo no path: {file_path}{file_name}_{pd.to_datetime('today').strftime('%Y_%m_%d')}.png")
    plt.show()
    
    return df_resultados_total

def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "█",
    print_end: str = "\r",
):
    """Chama um loop para criar uma barra de progresso no terminal

    Args:
        iteration (int): Iteração atual
        total (int): Total de iterações
        prefix (str): Prefixo que aparece na barra. Defaults to ""
        suffix (str): Sufixo que aparece na barra.  Defaults to ""
        decimals (int): Número de casas decimais.  Defaults to 1
        length (int): Tamanho da barra em caracteres. Defaults to 100
        fill (str): Texto que preenche a barra. Defaults to "█"
        print_end (str): Caracter final. Defaults to "\r"
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


"""
    
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
