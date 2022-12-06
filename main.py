import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression 
from nltk.stem.snowball import SnowballStemmer 
from sklearn.model_selection import train_test_split 
from nltk.tokenize import RegexpTokenizer  
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline 
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("phishing_site_urls.csv")
words_token = RegexpTokenizer(r'[A-Za-z]+')
data['text_tokenized'] = data.URL.map(lambda words: words_token.tokenize(words))
word_stemming = SnowballStemmer("english")
data['text_stemmed'] = data['text_tokenized'].map(lambda stem_word: [word_stemming.stem(word) for word in stem_word])
data['text_sent'] = data['text_stemmed'].map(lambda new_words: ' '.join(new_words))




cv=CountVectorizer()
feature=cv.fit_transform(data.text_sent)
feature[:5].toarray()
x_train,x_test,y_train,y_test=train_test_split(feature,data.Label)
logistic_model=LogisticRegression()
logistic_model.fit(x_train,y_train)
logistic_model.score(x_test,y_test)

"""The previous methods will be used to test and find the best ML model. KNN, Logistic regression
Naive bayes,Support vector Machines- add some more models. Pick the best and change in pipeline
"""

new_pipeline = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())
x_train, x_test, y_train, y_test = train_test_split(data.URL, data.Label)
new_pipeline.fit(x_train,y_train)
new_pipeline.score(x_test,y_test)
# new_model = pickle.load(open('phishing.pkl', 'rb'))
# final_result = new_model.score(x_test,y_test)
# print(final_result)

"""The above codes are used to create the phishing pickle file for creating pipelines
but those codes need not be executed coz we directly got the pickle file"""

links = ['anklavklasvas/yfuioaeh','youtube.com/']
#pickle.dump(new_pipeline,open('PHISH.pkl','wb')) #{Can create our own pickle}
model_2 = pickle.load(open('PHISH.pkl', 'rb'))
result = model_2.predict(links)
print(result)

"""The above code can be used to detect spam websites based on data"""

"""Pending work for this code,Check with other models. Add few more Features {Can be added from other codes, our own 
pkl file can be made}"""