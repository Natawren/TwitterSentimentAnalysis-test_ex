import pandas as pd
import numpy as np
import nltk
import re
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords

class TweetRecognition:
    
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stops = set(stopwords.words("english"))
        if os.path.exists(r"/home/qwerty/ex/model.pkl"):
            if os.path.isfile(r"/home/qwerty/ex/model.pkl") and os.path.getsize(r"/home/qwerty/ex/model.pkl") > 0:
                self.clf_file = 1
        else:
            self.clf_file = 0
        if os.path.exists(r"/home/qwerty/ex/vocab.pkl"):
            if os.path.isfile(r"/home/qwerty/ex/vocab.pkl") and os.path.getsize(r"/home/qwerty/ex/vocab.pkl") > 0:
                self.vocab_file = 1
        else:
             self.vocab_file = 0
                  
    def preprocessing_of_text(self, raw_text):
        #remove all urls:
        without_urls = re.sub(r"((www\.S+)|(http\S+))", "URL", raw_text)
        #remove numbers:
        without_numbers = re.sub(r"\d+", "", without_urls)
        #remove #
        without_oct = re.sub(r"#", "", without_numbers)
        #remove @
        without_at = re.sub(r"@\S+", "", without_oct)
        #make lower cases:
        words = without_at.lower().strip()
        # remove stop words
        stems = []
        # stemming
        for i in words.split():
            if i not in self.stops:
                stems.append(i)
        return " ".join(stems)
    
    def get_predict(self, text, flag=1):
        if  self.clf_file == 0 or self.vocab_file == 0:
            self.train_clf()
        with open(r"/home/qwerty/ex/model.pkl", 'rb') as f:
            clf = pickle.load(f)
        with open(r'/home/qwerty/ex/vocab.pkl', 'rb') as f1:
            vocab = pickle.load(f1)
        self.count_vect = CountVectorizer(vocabulary=vocab)
        if flag == 1:
            text = self.preprocessing_of_text(text)
            text = text.split()
        test_features_vect = self.count_vect.transform(text)
        predicted = clf.predict(test_features_vect)
        return (np.mean(predicted)) if flag == 1 else (predicted)
    
    def get_vectors(self, train_set):
        self.count_vect = CountVectorizer(analyzer = "word", 
                                     tokenizer = None, 
                                     preprocessor = None, 
                                     stop_words = None, 
                                     max_features = 5000)
        features_vect = self.count_vect.fit_transform(np.array(train_set.text))
        with open(r'/home/qwerty/ex/vocab.pkl', 'wb') as f:
            pickle.dump(self.count_vect.vocabulary_, f)
        return features_vect
    
    def train_clf(self):
        dataset = pd.read_csv(r"/home/qwerty/ex/training.csv", 
                              encoding='latin1', 
                              names=["target", "ids","date", "flag", "user", "text"])
        dataset.text = dataset.text.apply(lambda x: self.preprocessing_of_text(x))
        train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
        target = np.array(train_set.target)
        features = self.get_vectors(train_set)
        clf = LogisticRegression(C=1.).fit(features, target)
        with open(r"/home/qwerty/ex/model.pkl", 'wb') as f:
            pickle.dump(clf, f)
#       if you want to check a precision:
#         predicted = self.get_predict(np.array(test_set.text), 0)
#         print(np.mean(predicted == test_set.target))

# a = TweetRecognition()
# text = "it was really interesting, but on the other hand i wouldnt recommend it to my friends - too specific"
# a.train_clf()
# print(a.get_predict(text))
