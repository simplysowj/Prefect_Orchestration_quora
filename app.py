from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from prefect import task, flow
import distance
from fuzzywuzzy import fuzz

cv = CountVectorizer(max_features=3000)

#Load Data

def load_data(path: str, unwanted_cols: List) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.drop(unwanted_cols, axis=1, inplace=True)
    return data




#Text preprocessing

def preprocess(q: pd.Series) -> Any:
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q


def test_common_words(q1 : pd.Series,q2 : pd.Series) -> Any:
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)


def test_total_words(q1 : pd.Series,q2 : pd.Series) -> Any:
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))

def test_fetch_token_features(q1 : pd.Series,q2 : pd.Series) -> Any:
    SAFE_DIV = 0.0001

    STOP_WORDS = stopwords.words("english")


    token_features = [0.0] * 8

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    # Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))

    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


def test_fetch_length_features(q1 : pd.Series,q2 : pd.Series) -> Any:
    length_features = [0.0] * 3

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    # Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return length_features


def test_fetch_fuzzy_features(q1 : pd.Series,q2 : pd.Series) -> Any:
    fuzzy_features = [0.0] * 4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


   




def split_data(input_: pd.DataFrame, output_: pd.Series, test_data_ratio: float) -> Dict[str, Any]:
    X_tr, X_te, y_tr, y_te = train_test_split(input_, output_, test_size=test_data_ratio, random_state=0)
    return {'X_TRAIN': X_tr, 'Y_TRAIN': y_tr, 'X_TEST': X_te, 'Y_TEST': y_te}






def find_best_model(X_TRAIN: Any, Y_TRAIN: Any, estimator: Any) -> Any:
    # Enabling automatic MLflow logging for scikit-learn runs
    mlflow.sklearn.autolog(max_tuning_runs=None)
    with mlflow.start_run():        
        rf = RandomForestClassifier()
        rf.fit(X_TRAIN,Y_TRAIN)
        
        # Disabling autologging
        mlflow.sklearn.autolog(disable=True)
        
        return rf
# Workflow

def main(path: str):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Quora Duplicate Question Detector")
    # Define Parameters
    TARGET_COL = 'is_duplicate'
    UNWANTED_COLS = []
    TEST_DATA_RATIO = 0.2
    DATA_PATH = path
    
    # Load the Data
    dataframe = load_data(path=DATA_PATH, unwanted_cols=UNWANTED_COLS)
    dataframe = dataframe.sample(10,random_state=2)
    q1=dataframe['question1']
    q2=dataframe['question2']
    y=dataframe['is_duplicate']

    def query_point_creator(q1:Any,q2:Any,y:Any,i:Any) -> Any:
        input_query = []
        
        # preprocess
        q1 = preprocess(q1)
        q2 = preprocess(q2)
        input_query.append(i)
        input_query.append(y)
        # fetch basic features
        input_query.append(len(q1))
        input_query.append(len(q2))

        input_query.append(len(q1.split(" ")))
        input_query.append(len(q2.split(" ")))

        input_query.append(test_common_words(q1, q2))
        input_query.append(test_total_words(q1, q2))
        input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

        # fetch token features
        token_features = test_fetch_token_features(q1, q2)
        input_query.extend(token_features)

        # fetch length based features
        length_features = test_fetch_length_features(q1, q2)
        input_query.extend(length_features)

        # fetch fuzzy features
        fuzzy_features = test_fetch_fuzzy_features(q1, q2)
        input_query.extend(fuzzy_features)

        # bow feature for q1
        q1_bow = cv.fit_transform([q1]).toarray()

        # bow feature for q2
        q2_bow = cv.fit_transform([q2]).toarray()
        query = np.hstack((np.array(input_query).reshape(1, 24), q1_bow, q2_bow))
        #query.set_index(query.columns[0])
        return pd.DataFrame(query)
    
    
    final_df=dataframe.apply(lambda x: query_point_creator(x['question1'], x['question2'],x['is_duplicate'],x['id']), axis=1  )
    #final_df1 = load_data1(path1='C:\Users\Dhanamjaya\Desktop\Final_project\ML_orchestration\version-2\prefect_df.csv')
    #final_df=pd.DataFrame(final_df,columns=final_df[0,1:])
    print(final_df)
    
    final_df1=pd.read_csv('prefect_df.csv')
    # Identify Target Variable
     # Identify Target Variable
    target_data = final_df1['is_duplicate']
    input_data = final_df1.drop([TARGET_COL], axis=1)
    #input_data = final_df.iloc[:,1:].values
    print(input_data)
     
    #target_data= final_df.iloc[:,0].values
    print(target_data)

    # Split the Data into Train and Test
    train_test_dict = split_data(input_=input_data, output_=target_data, test_data_ratio=TEST_DATA_RATIO)
         




     # Model Training
    ESTIMATOR = RandomForestClassifier()
    
    X_TRAIN=train_test_dict['X_TRAIN']

    
    train_test_dict = split_data(input_=input_data, output_=target_data, test_data_ratio=TEST_DATA_RATIO)
    #print(X_train_transformed)
    classifier = find_best_model(train_test_dict['X_TRAIN'], train_test_dict['Y_TRAIN'], ESTIMATOR)
    y_pred = classifier.predict(train_test_dict['X_TEST'])
    accuracy_score(train_test_dict['Y_TEST'],y_pred)
    
    print(classifier.score(train_test_dict['X_TEST'], train_test_dict['Y_TEST']))




# Run the main function
main(path='./Data/train.csv')