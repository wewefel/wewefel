import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import random
import json
import pickle
import speech_recognition as sr
from gtts import gTTS
from VoiceAssistant import speak
from VoiceAssistant import get_audio

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

from Weather import weather



with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?" or "!" or "." or ","]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)



def chat():
    print("Say something (type quit to stop)")
    while True:

        print("")
        print("You: ")
        my_voice = get_audio()
        inp = str(my_voice)

        # "results" only has value of probability for each output. argmax gets output with highest probability
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]


        if inp.lower() == "quit":
            break

        elif inp.lower()[0:7] == "look up":
            PATH = "C:\chromedriver.exe"
            driver = webdriver.Chrome(PATH)

            driver.get("https://www.google.com/")

            search = driver.find_element_by_name("q")
            search.send_keys(inp[8:])
            search.send_keys(Keys.RETURN)
            top_text = driver.find_element_by_class_name("hgKElc")
            try:
                speak(top_text.text)
            finally:
                os.system("pause")

        elif tag == "weather":
            print("")
            print("Cum Slut Bot: ")
            print("What city?")
            speak("What city?")
            print("")
            print("Enter city name: ")
            my_voice2 = get_audio()
            city = my_voice2
            print(weather(city))
            cb_message = "The weather in" + city + "is" + weather(city)
            speak(cb_message)

        elif results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    message = random.choice(responses)
                    print("")
                    print("Cum Slut Bot: ")
                    print(message)
                    speak(message)
                    
        else:
            f1 = open("chatbot3.py", "r")
            f2 = open("unknown.py", "a+")
            f2.write(inp)
            f2.write("\n")
            f1.seek(0)
            f2.seek(0)
            print("")
            print("Wefel Bot: ")
            print("I don't understand. Try again")
            speak("I don't understand. Try again")

chat()
