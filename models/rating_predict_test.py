import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from models.preprocess import preprocess_text
from models.rating_predict import get_report, normalize_predictions, predict_with_model, train_and_evaluate_model, load_and_preprocess_data

@pytest.fixture
def sample_data():
    data = {
        "reviews.text": ["This product is great", "Terrible product", "Okay product", "Amazing product", "Bad product"],
        "reviews.rating": [5, 1, 3, 5, 2],
        "reviews.date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "name": ["Product1", "Product2", "Product3", "Product4", "Product5"]
    }
    df = pd.DataFrame(data)
    return df

def test_preprocess_text(sample_data):
    df = preprocess_text(sample_data.copy())
    assert df["reviews.text"].str.islower().all()

def test_normalize_predictions():
    pred = [6, 4, 2, 0, 3]
    normalized = normalize_predictions(pred)
    assert normalized == [5, 4, 2, 1, 3]

def test_load_and_preprocess_data():
    df = load_and_preprocess_data("data/reviews_v1_hiring_task.csv")
    assert not df.empty
    assert "reviews.text" in df.columns
    assert "reviews.rating" in df.columns

def test_train_and_evaluate_model(sample_data):
    X = sample_data["reviews.text"]
    y = sample_data["reviews.rating"]
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
    train_metrics, test_metrics = train_and_evaluate_model(X_train_idf, X_test_idf, y_train, y_test, log_model, "LinReg", models_metrics)
    
    assert isinstance(train_metrics, list)
    assert isinstance(test_metrics, list)

def test_predict_with_model(sample_data):
    X = sample_data["reviews.text"]
    y = sample_data["reviews.rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tfidf = TfidfVectorizer(min_df=0.001)
    X_train_idf = tfidf.fit_transform(X_train)
    X_test_idf = tfidf.transform(X_test)
    
    random_forest_model = RandomForestRegressor(max_features=200)
    random_forest_model.fit(X_train_idf, y_train)
    
    review = "This product exceeded my expectations. It's really amazing!"
    predicted_rating = predict_with_model(random_forest_model, tfidf.transform([review]))
    
    assert isinstance(predicted_rating, list)
    assert 1 <= predicted_rating[0] <= 5
