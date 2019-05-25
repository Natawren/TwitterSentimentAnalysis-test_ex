import pandas as pd
import numpy as np
import nltk
import re
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import SGDClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize
# from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer


def get_dataset():
    dataset = pd.read_csv('/home/qwerty/ex/training.csv', encoding='latin1', names=["target", "ids","date", "flag", "user", "text"])
    return dataset

def preprocessing_of_text(raw_text):
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
    #     tokens = nltk.word_tokenize(words)
    # remove stop words
    stems = []
    # stemming
    for i in words.split():
        if i not in stops:
#             stems.append(porter.stem(i))
              stems.append(i)
    # lancaster_stemmer = LancasterStemmer()
    # for i in result:
        # stems.append(lancaster_stemmer.stem(i))
        #stems.append(porter.stem(i))
    return " ".join(stems)

def train_clf(features, target, test_set):
#     clf = MultinomialNB().fit(features, target)
#     clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(features, target)
#     clf = RandomForestClassifier(n_estimators = 50).fit(features, target)
#     clf = GaussianNB().fit(features, target)

    clf = LogisticRegression(C=1.).fit(features, target)
    test_features_vect = count_vect.transform(np.array(test_set.text))
#     test_features_tf = tfidf_transformer.transform(test_features_vect)
#     test_features_tf = tfidf_transformer.transform(np.array(test_set.text))
    predicted = clf.predict(test_features_vect)
#     print(np.mean(predicted == test_set.target))
    with open('/home/qwerty/ex/model.pkl', 'wb') as f:
        pickle.dump(clf, f)

stops = set(stopwords.words("english"))
# porter = PorterStemmer()
dataset = get_dataset()
dataset.text = dataset.text.apply(lambda x: preprocessing_of_text(x))
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

count_vect = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
# tfidf_transformer = TfidfVectorizer(sublinear_tf=True, stop_words = "english")
# tfidf_transformer = TfidfTransformer()
features_vect = count_vect.fit_transform(np.array(train_set.text))
# features_tfidf = tfidf_transformer.fit_transform(features_vect)

# features_tfidf = tfidf_transformer.fit_transform(np.array(train_set.text))

target = np.array(train_set.target)
train_clf(features_vect, target, test_set)

