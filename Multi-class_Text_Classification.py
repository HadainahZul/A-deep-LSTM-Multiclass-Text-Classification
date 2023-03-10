# %% Import packages
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import re
import os

from modules import lstm_model_creation

# %%
# 1. Data Loading
URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

df = pd.read_csv(URL)

# %%
# 2. Data Inspection
df.describe()
df.info()
df.head()

# %%
# 99 duplicated data here
df.duplicated().sum()

# To check NaN
df.isna().sum()

# %%
print(df['text'][0])

# %%
# 3. Data Cleaning #regex = Regular Expression

# a. Remove numbers --> Settled
# b. Remove HTML Tags --> No HTML Tags
# c. Remove punctuation --> Settled
# d. $number and special characters and punctuations --> settled

for index, data in enumerate(df['text']):
    df['text'][index] = re.sub('<.*?>', '', data)
    df['text'][index] = re.sub('[^a-zA-Z]', ' ', df['text'][index]).lower()

# %%
# 4. Feature Selection (X --> text, y --> category)
text = df['text']
category = df['category']

# %%
#5. Data Preprocessing

# Tokenizer
#num_words = 3000
num_words = 5000 # unique number of words in all sentences
oov_token = '<OOV>' # out of vocab

tokenizer = Tokenizer(num_words = num_words, oov_token = oov_token)

# to train the tokenizer --> mms.fit()
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# to transform the text using tokenizer --> mms.transform
text = tokenizer.texts_to_sequences(text)

# Padding
#padded_text = pad_sequences(text, maxlen = 200, padding='post', truncating='pre')
padded_text = pad_sequences(
    text, maxlen = 200, padding='post', truncating='post')

# %%
#OneHotEncoder
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(category[::,None])

# %%
# Do the Train Test Split
# Expand dimension before feeding to train_test_split
padded_text = np.expand_dims(padded_text, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(padded_text, category, test_size = 0.2, random_state = 123)

# %%
# Model Development
# embedding_layer = 64
# model = Sequential()
# model.add(Embedding(num_words, embedding_layer))
# model.add(LSTM(embedding_layer, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(64))
# model.add(Dropout(0.5))
# model.add(Dense(5, activation = 'softmax'))
# model.summary()
# plot_model(model, show_shapes=True)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model = lstm_model_creation(num_words, category.shape[1], dropout=0.5)

#hist = model.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size = 64, epochs = 10)

# %%
# Tensorboard callback
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d - %H%M%S'))

ts_callback = TensorBoard(log_dir = LOGS_PATH)
#es_callback = EarlyStopping(monitor = 'val_los', patience = 5, verbose = 0, restore_best_weights = True)

#Train model
hist = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs = 12, batch_size = 64, callbacks = [ts_callback])

# %%
#Plot the graph Training and Validation
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'])
plt.show()

y_predicted = model.predict(X_test)

# %%
# Model Analysis
y_predicted = model.predict(X_test)
y_predicted = np.argmax(y_predicted, axis = 1)
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted)

print(classification_report(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))

# %% 
# Model Saving

# To save trained model
model.save('model.h5')

# To save one hot encoder
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

# %%
# To save tokenizer
token_jason = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(token_jason,f)

# %%
