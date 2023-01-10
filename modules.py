# %%
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
import re

# %%
def lstm_model_creation(num_words, nb_classes, embedding_layer = 64, dropout=0.5, num_neurons=64):
   
    model = Sequential()
    model.add(Embedding(num_words, embedding_layer))
    model.add(LSTM(embedding_layer, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(num_neurons))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes, activation = 'softmax'))
    model.summary()
    plot_model(model, show_shapes=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return model

# %%
