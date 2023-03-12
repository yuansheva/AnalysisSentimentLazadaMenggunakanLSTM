import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import streamlit as st

file_dataset = 'lazada-reviews.csv'
df = pd.read_csv(file_dataset)

df = df[['category','itemId','reviewContent','rating']]
df = df.dropna(subset=['reviewContent'])

label=[]
for index, row in df.iterrows():
    if row['rating']>=4:
        label.append(1)
    else:
        label.append(0)

df['label']=label
df=df.drop(columns='rating', axis=1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(df['reviewContent'], df['label'], \
                                                    test_size=0.2, random_state=0)

print('Load %d training examples and %d validation examples. \n' %(X_train.shape[0],X_test.shape[0]))
print('Show a review in the training set : \n', X_train.iloc[10])

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

checkpoint = ModelCheckpoint("LSTM.h5", monitor='val_loss', save_freq='epoch', save_best_only=True)
model.load_weights("LSTM.h5")

# Menambahkan tampilan aplikasi web dengan Streamlit
st.title('Aplikasi Prediksi dengan Model LSTM')
st.write('Masukkan Text berikut untuk melakukan prediksi:')

inputPrediksi = st.text_input("Input Teks Komentar Barang")
if st.button('Prediksi'):
    # load model from single file
    model = load_model('LSTM.h5')

    inputHasil = [inputPrediksi]
    tokenized_test = tokenizer.texts_to_sequences(inputHasil)
    X_test = pad_sequences(tokenized_test, maxlen=max_review_length)
    prediction = model.predict(X_test)
    print(prediction)
    pred_labels = []
    if prediction > 0.5:
        s = 'Positif'
        pred_labels.append(1)
    else:
        s = 'Negatif'
        pred_labels.append(0)

    st.write("Prediksi Sentiment Barang :" , s)
