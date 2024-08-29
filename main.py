import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv('db.csv', sep=';', header=None)
df.head()
pass

# Seleciona as colunas 'genras' e 'lyrics'
genras = df.iloc[:, 2]
lyrics = df.iloc[:, 3]

# Filtra as linhas onde há valores NaN em 'genras' ou 'lyrics'
df = df.dropna(subset=[df.columns[2], df.columns[3]])

# Atualiza os valores de 'genras' e 'lyrics' com os valores sem NaNs
genras = df.iloc[:, 2].values
lyrics = df.iloc[:, 3].values

# Dropa todas as colunas após a coluna de 'lyrics'
df = df.drop(columns=df.columns[4:])

print(len(genras), len(lyrics))


# 1. Treinar Word2Vec
sentences = [lyric.split() for lyric in lyrics]  # Divide as letras em palavras
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 2. Transformar as letras em sequências de embeddings
def lyrics_to_embedding_sequences(lyrics, word2vec_model, max_seq_length):
    embedding_sequences = []
    for lyric in lyrics:
        words = lyric.split()
        embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        embedding_sequences.append(embeddings)
    return pad_sequences(embedding_sequences, maxlen=max_seq_length, dtype='float32', padding='post', truncating='post')

max_seq_length = 500  # Limitar as sequências a 500 palavras
X = lyrics_to_embedding_sequences(lyrics, word2vec_model, max_seq_length)

# 3. Codificar as classes e dividir os dados
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(genras)
num_classes = len(np.unique(y))  # Número de classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Construir e treinar o modelo LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(max_seq_length, 100)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Softmax para multi-classes

#model.compile(optimizer=Adam(learning_rate=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer=Nadam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Treinamento do modelo
model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_split=0.2, callbacks=[early_stopping])


# 5. Avaliação no conjunto de teste
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test accuracy: {accuracy}')

# Função para plotar a contagem de gêneros musicais
def plot_genre_counts():
    # Contar a quantidade de cada gênero
    contagem_genras = pd.Series(genras).value_counts()
    contagem_genras = contagem_genras.drop(contagem_genras.index[-1])  # Remover último gênero se necessário
    
    # Plotar o resultado
    plt.figure(figsize=(10, 6))
    num_barras = len(contagem_genras)
    cores = plt.cm.viridis(np.linspace(0, 1, num_barras))
    ax = contagem_genras.plot(kind='bar', color=cores)
    
    # Adicionar título e rótulos
    plt.title('Contagem de Gêneros Musicais', fontsize=16)
    plt.xlabel('Gênero', fontsize=14)
    plt.ylabel('Contagem', fontsize=14)
    
    # Adicionar valores acima das barras
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.0, height, int(height), ha='center', va='bottom', fontsize=12)

    # Mostrar o gráfico
    #plt.show()
    plt.pause(2)  # Exibe o gráfico por 2 segundos
    plt.close()

# Exemplo de uso
plot_genre_counts()

# Função para exibir os resultados
def results():
    plot_genre_counts()

def main():
    results()

if __name__ == '__main__':
    main()