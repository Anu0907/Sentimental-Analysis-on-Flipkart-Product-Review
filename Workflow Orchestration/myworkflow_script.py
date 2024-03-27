import pandas as pd
from sklearn.model_selection import train_test_split

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from prefect import task, flow

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    X = data[inputs]
    y = data[output]
    return X, y

@task
def split_train_test(X, y, test_size=0.25, random_state=0):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@task
def clean(doc):
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = re.sub(r"[^a-zA-Z]", " ", doc)
    doc = re.sub(r'\W+', ' ', doc)
    doc = doc.translate(str.maketrans('', '', string.punctuation))

    doc = doc.lower()

    tokens = nltk.word_tokenize(doc)

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    stop_words = set(stopwords.words('english'))

    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]

    return " ".join(filtered_tokens)

@task
def feature_extraction(X_train, X_test, y_train, y_test):
    """
    Feature Extraction of the data.
    """
    vect = CountVectorizer(preprocessor=clean)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    return X_train_dtm, X_test_dtm, y_train, y_test

@task
def train_model(X_train_dtm, y_train, hyperparameters):
    """
    Training the machine learning model.
    """
    clf = KNeighborsClassifier(**hyperparameters)
    clf.fit(X_train_dtm, y_train)
    return clf

@task
def evaluate_model(model, X_train_dtm, y_train, X_test_dtm, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(X_train_dtm)
    y_test_pred = model.predict(X_test_dtm)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score

# Workflow
@flow(name="KNN Training Flow")
def workflow():
    DATA_PATH = "batminton_data.csv"
    INPUTS = 'review_text'
    OUTPUT = 'sentiment'
    HYPERPARAMETERS = {'n_neighbors': 15, 'p': 2}
        
    # Load data
    df_batminton = load_data(DATA_PATH)

    # Identify Inputs and Output
    X, y = split_inputs_output(df_batminton, INPUTS, OUTPUT)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Preprocess the data
    X_train_dtm, X_test_dtm, y_train, y_test = feature_extraction(X_train, X_test, y_train, y_test)

    # Build a model
    model = train_model(X_train_dtm, y_train, HYPERPARAMETERS)
    
    # Evaluation
    train_score, test_score = evaluate_model(model, X_train_dtm, y_train, X_test_dtm, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)


if __name__ == "__main__":
    workflow.serve(
        name="my-first-deployment",
        cron="* * * * *"
    )
