import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
import string

import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import logging

from models.preprocess import preprocess_text

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import logging

def preprocess_text(df):
    # Dummy preprocess function, replace with actual preprocessing steps
    df['reviews.text'] = df['reviews.text'].apply(lambda x: x.lower())
    return df

def get_report(test_data, predicted_data, model_name="", models_metrics=None):
    metrics = [
        mean_squared_error(test_data, predicted_data, squared=False),
        r2_score(test_data, predicted_data),
        accuracy_score(test_data, predicted_data),
    ]
    if model_name and models_metrics is not None:
        models_metrics.at[model_name, "RMSE"] = metrics[0]
        models_metrics.at[model_name, "Accuracy"] = metrics[2]
        models_metrics.at[model_name, "R^2"] = metrics[1]
    return metrics

def normalize_predictions(pred):
    new_pred = [i if i <= 5 else 5 for i in pred]
    new_pred = [i if i >= 1 else 1 for i in new_pred]
    new_pred = [round(i) for i in new_pred]
    logging.info("Normalizing predictions")
    return new_pred

def predict_with_model(model, X_test):
    logging.info("Starting prediction with model.")
    predictions = model.predict(X_test)
    normalized_predictions = normalize_predictions(predictions)
    return normalized_predictions

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name, models_metrics):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_metrics = get_report(y_train, train_pred)
    test_metrics = get_report(y_test, test_pred, model_name, models_metrics)
    return train_metrics, test_metrics

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, sep=",", encoding="latin-1").sample(500)
    data = preprocess_text(data)
    df = data[["reviews.text", "reviews.rating", "reviews.date", "name"]]
    return df

def main():
    df = load_and_preprocess_data("data/reviews_v1_hiring_task.csv")
    X = df["reviews.text"]
    y = df["reviews.rating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf = TfidfVectorizer(min_df=0.001)
    X_train_idf = tfidf.fit_transform(X_train)
    X_test_idf = tfidf.transform(X_test)

    models_metrics = pd.DataFrame({
        "Name": ["LinReg", "DesTree", "RanFor"],
        "Accuracy": [0, 0, 0],
        "RMSE": [0, 0, 0],
        "R^2": [0, 0, 0],
    }).set_index("Name")

    log_model = LogisticRegression(max_iter=1000)
    train_and_evaluate_model(X_train_idf, X_test_idf, y_train, y_test, log_model, "LinReg", models_metrics)

    tree_model = DecisionTreeRegressor(max_depth=20)
    train_and_evaluate_model(X_train_idf, X_test_idf, y_train, y_test, tree_model, "DesTree", models_metrics)

    random_forest_model = RandomForestRegressor(max_features=200)
    train_and_evaluate_model(X_train_idf, X_test_idf, y_train, y_test, random_forest_model, "RanFor", models_metrics)

    print(models_metrics)

    review = "This product exceeded my expectations. It's really amazing!"
    predicted_rating = predict_with_model(random_forest_model, tfidf.transform([review]))
    print(f"Predicted Rating: {predicted_rating}")

if __name__ == "__main__":
    main()
