import pandas as pd 
import sklearn
import numpy as np
import nltk
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn import tree
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support


import gensim, logging
from gensim.models import Word2Vec
from scipy import sparse

	def loadData(filePath="dataset.csv"):
	    data=pd.read_csv("/home/administrator/data/categories-data/Train-Data/fps-with-cat-train.csv")
	    data['CategoryFB'] = data['CategoryFB'].fillna(data['CategoryV2'])
	    data['Description'] = data['Description'].fillna(data['Name'])
	    return data["Tag"],data["Name"],data["CategoryV2"]

def preProcessing(features):
    num_descs = features.size
    clean_wordlist = []
    clean_descs = []
    stops = set(stopwords.words('english'))
    #letters_only = []
    for i in range( 0, num_descs):
        #letters_only = re.sub("[^a-zA-Z]", " ", features[i]) 
        words = features[i].lower().split()
        words = [w.lower() for w in words if not w in stops]  
        clean_wordlist.append(words)
        clean_descs.append(" ".join(words))
    return clean_descs, clean_wordlist


def getDTMByTFIDF(features,nfeatures):
    tfIdf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english', max_features=nfeatures)
    dtm = tfIdf_vectorizer.fit_transform(features).toarray()
    return dtm,tfIdf_vectorizer

def featuresByChiSq(features,labels,nFeature=5000):
    chi2_model = SelectKBest(chi2,k=nFeature)
    dtm = chi2_model.fit_transform(features,labels)
    return dtm,chi2_model

def featuresByInformationGain(features,labels):
    treeCL = tree.DecisionTreeClassifier(criterion="entropy")
    treeCL = treeCL.fit(features,labels)
    transformed_features = SelectFromModel(treeCL,prefit=True).transform(features)
    return transformed_features

def featuresByLSA(features,ncomponents=100):
    svd = TruncatedSVD(n_components=ncomponents)
    normalizer =  Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    dtm_lsa = lsa.fit_transform(features)
    return dtm_lsa

def makeFeatureVec(words, model, num_features):
    feature_vec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec,model[word]) 

    feature_vec = np.divide(feature_vec,nwords)
   
    return feature_vec

def getAvgFeatureVecs(title, model, num_features):
    counter = 0.
    titleFeatureVecs = np.zeros((len(title), num_features),dtype="float32")
    for t in title:
        titleFeatureVecs[counter] = makeFeatureVec(t, model,num_features)
        counter = counter + 1.
    return titleFeatureVecs


def crossValidate(document_term_matrix,labels,classifier="SVM",nfold=10):
    clf = None
    precision = []
    recall = []
    fscore = []
    
    if classifier == "RF":
        clf = RandomForestClassifier()
    elif classifier == "NB":
        clf = MultinomialNB()
    elif classifier == "SVM":
        clf = LinearSVC()
    
    skf = StratifiedKFold(labels, n_folds=nfold)

    for train_index, test_index in skf:
        X_train, X_test = document_term_matrix[train_index], document_term_matrix[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        p,r,f,s = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        
    return round(np.mean(precision),3),round(np.mean(recall),3),round(np.mean(fscore),3)

tags, descs, labels = loadData()
processed_descs, processed_descs_wordlist = preProcessing(descs)
dtm,vect = getDTMByTFIDF(processed_descs,2000)

chisqDtm, chisqModel = featuresByChiSq(dtm,labels,2000)

features = featuresByInformationGain(dtm, labels)

precision, recall, fscore = crossValidate(chisqDtm,labels,"SVM",10)

print precision, recall, precision_recall_fscore_support

from sklearn.cross_validation import train_test_split

train_descs, test_descs = train_test_split(descs, test_size=0.1, 
                                           random_state=42)

train_descs, test_descs = train_test_split(descs, test_size=0.1, 

train_labels, test_labels = train_test_split(labels, test_size=0.1, 
                                           random_state=42)

train_descs = train_descs.reset_index(drop=True)
test_descs = test_descs.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
test_labels = test_labels.reset_index(drop=True)

processed__train_descs, processed_train_descs_wordlist = preProcessing(train_descs)
processed_test_descs, processed_test_descs_wordlist = preProcessing(test_descs)

dtm_train,vect_train = getDTMByTFIDF(processed__train_descs,2000)
chisqDtmTrain, chisqModelTrain = featuresByChiSq(dtm_train,train_labels,2000)

clf = LinearSVC()

model = clf.fit(chisqDtmTrain, train_labels)
dtm_test,vect_test = getDTMByTFIDF(processed_test_descs,2000)
chisqDtmTest, chisqModelTest = featuresByChiSq(dtm_test,test_labels,2000)
y_pred = model.predict(chisqDtmTest)

p,r,f,s = precision_recall_fscore_support(test_labels, y_pred, average='weighted')

