import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import Sastrawi
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string, re
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix
import pickle
from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import streamlit as st

# Import data

file_dataset = 'lazada-reviews.csv'
df = pd.read_csv(file_dataset)

df = df[['category','itemId','reviewContent','rating']]
df = df.dropna(subset=['reviewContent'])
df

df['rating'].value_counts().sort_values(ascending=True)

print("Summary statistics of numerical features : \n", df.describe())
print("=======================================================================")
print("\nTotal number of reviews: ",len(df))
print("=======================================================================")
print("\nTotal number of category: ", len(list(set(df['category']))))
print("=======================================================================")
print("\nTotal number of unique products: ", len(list(set(df['itemId']))))
print("=======================================================================")

print("\nPercentage of reviews with positive sentiment : {:.2f}%"\
      .format(df[df['rating']>=4]["reviewContent"].count()/len(df)*100))
print("=======================================================================")

print("\nPercentage of reviews with negative sentiment : {:.2f}%"\
      .format(df[df['rating']<=3]["reviewContent"].count()/len(df)*100))
print("=======================================================================")

label=[]
for index, row in df.iterrows():
    if row['rating']>=4:
        label.append(1)
    else:
        label.append(0)

df['label']=label
df=df.drop(columns='rating', axis=1)
df

print(df.shape)
print(df.label.value_counts(normalize=True))

# Wordcloud
#polarity = 0
# trains1=df[df['label']==0]
# all_text=' '.join(word for word in trains1['reviewContent'])
# word_cloud=WordCloud(colormap='Reds', width=1000, height=1000,mode='RGBA',background_color='white').generate(all_text)
# plt.figure(figsize=(20,20))
# plt.imshow(word_cloud, interpolation='bilinear')
# plt.axis('off')
# plt.margins(x=0, y=0)
# plt.show()
# #polarity = 1
# trains1=df[df['label']==1]
# all_text=' '.join(word for word in trains1['reviewContent'])
# word_cloud=WordCloud(colormap='Blues', width=1000, height=1000,mode='RGBA',background_color='white').generate(all_text)
# plt.figure(figsize=(20,20))
# plt.imshow(word_cloud, interpolation='bilinear')
# plt.axis('off')
# plt.margins(x=0, y=0)
# plt.show()


# Data preprocessing
def cleansing(df):
    #lowertext
    df=df.lower()
    
    #Remove Punctuation
    remove=string.punctuation
    translator=str.maketrans(remove,' '*len(remove))
    df=df.translate(translator)
    
    #Remove ASCII & UNICODE
    df=df.encode('ascii','ignore').decode('utf-8')
    df=re.sub(r'[^\x00-\x7f]',r'', df)
    
    #Remove Newline
    df=df.replace('\n',' ')
    
    return df
review = []
for index, row  in df.iterrows():
    review.append(cleansing(row['reviewContent']))
review


# Data preparation
# factory = StopWordRemoverFactory()
# stopword = factory.create_stop_word_remover()
 
# # Contoh
# sentence = 'alhamdulillah  sampai dengan selamat   smoga awet   terima kasih lazada retail coocaa '
# stop = stopword.remove(sentence)
# print(stop)

# review = []
# for index, row in df.iterrows():
#     review.append(stopword.remove(row['reviewContent']))


# df['reviewContent']=review
# df

# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

# # contoh
# sentence = 'packing rapi menggunakan kayu  mantap   dapet headphone'
# s_clean = stemmer.stem(sentence)
 
# print(s_clean)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(df['reviewContent'], df['label'], \
                                                    test_size=0.2, random_state=0)

print('Load %d training examples and %d validation examples. \n' %(X_train.shape[0],X_test.shape[0]))
print('Show a review in the training set : \n', X_train.iloc[10])
X_train,y_train


# Model LSTM
train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)
print("Training data size : ", train_df.shape)
print("Validation data size : ", test_df.shape)

top_words = 20000
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(train_df['reviewContent'])
list_tokenized_train = tokenizer.texts_to_sequences(train_df['reviewContent'])

max_review_length = 200
X_train = pad_sequences(list_tokenized_train, maxlen=max_review_length)
y_train = train_df['label']

embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train,y_train, epochs=15, batch_size=256, validation_split=0.2)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b--', label='loss')
plt.plot(history.history['val_loss'], 'r:', label='val_loss')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b--', label='acc')
plt.plot(history.history['val_accuracy'], 'r:', label='val_acc')
plt.xlabel('Epochs')
plt.legend()

plt.show()

list_tokenized_test = tokenizer.texts_to_sequences(test_df['reviewContent'])
X_test = pad_sequences(list_tokenized_test, maxlen=max_review_length)
y_test = test_df['label']
prediction = model.predict(X_test)
y_pred = (prediction > 0.5)
print("Accuracy of the model : ", accuracy_score(y_pred, y_test))
print('F1-score: ', f1_score(y_pred, y_test))
print('Confusion matrix:')
confusion_matrix(y_test,y_pred)


# simpan model
model.save('LSTM.h5')

sentences = ['Bagus, sesuai foto', 'paket buruk']
# load model from single file
model = load_model('LSTM.h5')

tokenized_test = tokenizer.texts_to_sequences(sentences)
X_test = pad_sequences(tokenized_test, maxlen=max_review_length)
prediction = model.predict(X_test)
print(prediction)
pred_labels = []
for i in prediction:
    if i > 0.5:
        pred_labels.append(1)
    else:
        pred_labels.append(0)
        
for i in range(len(sentences)):
    print(sentences[i])
    if pred_labels[i] == 1:
        s = 'Positif'
    else:
        s = 'Negatif'
    print("Prediksi sentiment : ",s)