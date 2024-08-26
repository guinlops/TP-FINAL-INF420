import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
import matplotlib.pyplot as plt

# Carregar os dados
# try:
#     df = pd.read_csv('db.csv', on_bad_lines='skip')
# except():
#     print("Nao foi possivel abrir o banco de dados")
# # Codificação dos rótulos (gêneros)
# #label_encoder = LabelEncoder()
# #y = label_encoder.fit_transform(df['genra'])
# Y = df.iloc[:, 1].values   # Segunda coluna (label) Koi Disposition, Confirmed, False Positive
# print(Y)

df = pd.read_csv('db.csv', sep=';', header=None)

genras = df.iloc[:, 2].values
df = df.drop(columns=df.columns[4:]) # Dropa todas as colunas depois das colunas lyrics
genras = df.iloc[:, 2].dropna() #dropa todas as linhas em que a valores Not available



# Ler o CSV com o separador ";"
# df = pd.read_csv('db.csv', sep=';', quoting=3)  # quoting=3 desativa o tratamento de aspas

def plot_genre_counts():
    # Ignorar a primeira linha e pegar a coluna de gêneros
   
    
    # Contar a quantidade de cada gênero
    contagem_genras = genras.value_counts()
    contagem_genras = contagem_genras.drop(contagem_genras.index[-1])
    # Plotar o resultado
    plt.figure(figsize=(10, 6))
    #ax=contagem_genras.plot(kind='bar', color='skyblue')
    
    # Adicionar título e rótulos
    num_barras = len(contagem_genras)
    cores = plt.cm.viridis(np.linspace(0, 1, num_barras))
    ax = contagem_genras.plot(kind='bar', color=cores)
    plt.title('Contagem de Gêneros Musicais', fontsize=16)
    plt.xlabel('Gênero', fontsize=14)
    plt.ylabel('Contagem', fontsize=14)
    
    for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.0, height, int(height), ha='center', va='bottom', fontsize=12)

    # Mostrar o gráfico
    plt.show()

# Exemplo de uso:



def results():
     plot_genre_counts()

def main():
  

    results()
    
   
    
    

if __name__ == '__main__':
    main()
