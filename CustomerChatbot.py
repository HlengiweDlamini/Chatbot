import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
import json
import pickle

# json file with training data
with open("intents.json") as file:
    data = json.load(file)

# save variables in pickle file & load lists
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words =[]
    labels = []
    docs_x = [] # list of all patterns
    docs_y = [] # corresponding entries in x & y
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    # remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    
    labels = sorted(labels)
    
    # training model with bag of words(a list of length of words with positions of occurrence)
    training = []
    output =[]
    out_empty = [0 for _ in range(len(labels))]
    
    # creating bag of words
    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc]
        
        # go through all diff words in document
        for w in words:
            if w in wrds: # if the word exists in current pattern
                bag.append(1) # put 1 if it exixts
            else:
                bag.append(0)
        # generate output
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    # turn lists into NP arrays to feed them into model
    training = np.array(training)
    output = np.array(output)
    
    # write variables into pickle file so we can save it
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


# to train the model
# add this connected layer(connected to inputs) to neural network with 8 neurons for 1st hidden layer
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# load the model(wont train model if one already exists)
try:
    model.load("model.h5")
except:
    # pass model with all training data || epoch= amount of times model will see the same data
    model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
    # save model
    model.save("model.h5")

# -----------------------------making predictions---------------------------------------

# turn sentence input from user into bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))] # store all words in blank list
    
    # list of tokenized words
    s_words = nltk.word_tokenize(s)
    # stem words
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    # loop to generate bag list properly
    for se in s_words:
        for i, w in enumerate(words):
            if w == se: # current word in list = word in sentence
                bag[i] = 1
    
    # convert bag of words to numpy array and return    
    return np.array(bag)

# ask user for sentence then give response
def chat():
    print("start talking woith the bot (type quit to stop)!")
    while True:
        inp = input("You: ") # user identification when typing
        if inp.lower() == "quit": # end program when user types quit
            break
        
        # if user doesnt quit
        results = model.predict([bag_of_words(inp, words)])
        # gives index of greatest probability value in list
        results_index = np.argmax(results)
        
        # use index to figure out which response to display
        tag = labels[results_index]
        
        # open json file, find tag and pick random response
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                
        print(random.choice(responses))
         
chat()