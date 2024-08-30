import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, BatchNormalization, SpatialDropout1D, Flatten
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import HeNormal
import matplotlib.pyplot as plt

# 1. Carregar o arquivo CSV sem cabeçalho, separando as colunas por ';'
df = pd.read_csv('db.csv', sep=';', header=None)

# 2. Remover colunas da 5ª em diante (indexadas como 4 e superiores)
df = df.drop(columns=df.columns[4:])

# 3. Remover a primeira linha do DataFrame
df = df.drop(df.index[:1])

# 4. Nomear as colunas do DataFrame
df.columns = ['Music', 'Artist', 'Genra', 'Lyrics']

# 5. Contar a frequência de cada valor único na coluna 'Genra'
df['Genra'].value_counts()

# 6. Remover linhas onde a coluna 'Genra' tem o valor 'pop'
df = df[df['Genra'] != 'pop']

# 7. Contar o número de classes (gêneros musicais) após a remoção das linhas
classes = len(df['Genra'].value_counts())

# 8. Dividir a coluna 'Lyrics' em listas de palavras
df['Words'] = df['Lyrics'].str.split()

# 9. Contar o número de palavras diferentes em cada letra de música
df['Diferent_words'] = df['Words'].str.len()

# 10. Remover linhas com valores ausentes (NaN)
df.dropna(inplace=True)

# 11. Definir hiperparâmetros para o modelo de processamento de texto
# Parâmetros do modelo
MAX_WORDS = 2500
MAX_SEQUENCE_LENGTH = 355
EMBEDDING_DIM = 100
GRADIENT_CLIP_VALUE = 0.1

# 12. Inicializar o Tokenizer, que converterá o texto em sequências de números inteiros
tokenizer = Tokenizer(num_words=MAX_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')

# 13. Ajustar o Tokenizer no texto das letras de música
tokenizer.fit_on_texts(df.Lyrics.values)

# 14. Obter o índice de palavras do Tokenizer (mapeamento palavra -> número)
word_index = tokenizer.word_index

# 15. Converter as letras em sequências numéricas com base no Tokenizer
X = tokenizer.texts_to_sequences(df.Lyrics.values)

# 16. Preencher (ou truncar) as sequências para que todas tenham o mesmo comprimento
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# 17. Converter as classes (gêneros musicais) em uma matriz binária (one-hot encoding)
Y = pd.get_dummies(df['Genra']).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# Construindo o modelo
model = Sequential()
model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2, kernel_initializer=HeNormal()))
model.add(Dense(len(df['Genra'].value_counts()), activation='softmax'))

# Reduzindo a taxa de aprendizado para maior estabilidade
optimizer = Adam(learning_rate=0.000001, clipvalue=GRADIENT_CLIP_VALUE)

# Compilando o modelo
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.9), optimizer=optimizer, metrics=['accuracy'])

# Treinamento
epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Avaliação
accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

# 28. Testar o modelo com novos textos
X_test=[
    "There's a lady who's sure All that glitters is gold And she's buying a stairway to heaven...", 
    "Uh, Uh, Uh, 1, 2, 1, 2 Uh, Uh, 1, 2, 1, 2, uh, uh All my dogs It's bigger than hip hop..."]

# 29. Converter os novos textos em sequências numéricas
X_test=tokenizer.texts_to_sequences(X_test)

# 30. Preencher/truncar as sequências para que tenham o mesmo comprimento
X_test=pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

# 31. Fazer previsões com o modelo nos novos textos
model.predict(X_test)


result = model.predict(X_test)

print(result)