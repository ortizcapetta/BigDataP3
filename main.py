import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import pandas as pd

'''#########-------TRAINING-------########'''

training_data = 'Data/TrainingData.csv'
num_labels = 3 #0,1 or 2
batch_size = 50
vocab_size = 2500
tokenizer = Tokenizer(num_words = vocab_size)

trdata = pd.read_csv("Data/TrainingData.csv",header=None, names=['tweet', 'sentiment'])
trsize = int(len(training_data) * .8)
train_texts = trdata['tweet'][:trsize]
train_tags = trdata['sentiment'][:trsize]
test_texts = trdata['tweet'][trsize:]
test_tags = trdata['sentiment'][trsize:]


tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_texts)

x_train = tokenizer.texts_to_matrix(train_texts, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_texts, mode='tfidf')

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

encoder = LabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

#FIRST MODEL

model = Sequential()
model.add(Dense(700, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(700))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)


print('Test score:', score[0])
print('Test accuracy:' + str(score[1] *100) + "%")

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

#SECOND MODEL

model2 = Sequential()
model2.add(Dense(512, input_shape=(vocab_size,), activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(256, activation='tanh'))
model2.add(Dropout(0.5))
model2.add(Dense(num_labels, activation='softmax'))

model2.compile(loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])

model2.fit(x_train, y_train,
  batch_size=32,
  epochs=5,
  verbose=1,
  validation_split=0.1,
  shuffle=True)

score2 = model2.evaluate(x_test, y_test,batch_size=batch_size, verbose=1)
print('Test score:', score2[0])
print('Test accuracy:' + str(score2[1] *100) + "%")



'''#####################------PREDICTING---------###################'''
tweets = '/home/alejandra/Documents/CIIC5995/BigDataP2/results/keywords/keywords_alltext/part-00000.csv'

lines = [line.rstrip('\n') for line in open(tweets)]
flattweets =[]


for x in lines:
    flattweets.append(re.sub("[^0-9a-zA-Z]", " ",x))
with open('Data/tweets_flat.csv','w') as f:
    for item in flattweets:
        f.write("%s\n"%item)


full_tweets = pd.read_csv("Data/tweets_flat.csv",header=None, names=['tweet'])
keytweets = tokenizer.texts_to_matrix(full_tweets['tweet'])

text_labels = encoder.classes_

with open('Results/model1res.csv', 'w') as m1:
    with open('Results/model2res.csv','w') as m2:
        for i in range(len(flattweets)):
            prediction1 = model.predict(np.array([keytweets[i]]))
            predicted_label1 = text_labels[np.argmax(prediction1[0])]
            prediction2 = model2.predict(np.array([keytweets[i]]))
            predicted_label2 = text_labels[np.argmax(prediction2[0])]

            m1.write("%s,%s\n"%(str(flattweets[i]), str(predicted_label1)))
            m2.write("%s,%s\n" % (str(flattweets[i]), str(predicted_label2)))







