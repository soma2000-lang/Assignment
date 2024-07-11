import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
import string
import seaborn as sns
import nltk 
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


from models.preprocess import preprocess_text
data    = pd.read_csv('data/reviews_v1_hiring_task.csv', sep=',', encoding='latin-1').sample(500)
data=preprocess_text(data)
df = data[['reviews.text','reviews.rating','reviews.date','name','manufacturer']]
print(df) #getting the clean data

X=df['reviews.text']
y=df['reviews.rating']
models_metrics = pd.DataFrame({'Name':['LinReg', 'DesTree', 'RanFor'],
                               'Accuracy':[0, 0, 0],
                               'RMSE':[0, 0, 0],
                               'R^2':[0, 0, 0]})

models_metrics.set_index('Name', inplace=True)
def get_report(test_data, predicted_data, model_name=''):
    metrics = [mean_squared_error(test_data, predicted_data, squared=False),
              r2_score(test_data, predicted_data),
              accuracy_score(test_data, predicted_data)]
    if model_name:
        models_metrics.at[model_name, 'RMSE'] = metrics[0]
        models_metrics.at[model_name, 'Accuracy'] = metrics[2]
        models_metrics.at[model_name, 'R^2'] = metrics[1]
    print(f'The RMSE on data is: {metrics[0]}')
    print(f'The R2 on data is: {metrics[1]}')
    print(f'Accuracy is: {metrics[2]}')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Apply one-hot encoding only to the feature variables
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align columns of X_test to match X_train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Ensure y_train and y_test are 1-dimensional
y_train = y_train if y_train.ndim == 1 else y_train.iloc[:, 0]
y_test = y_test if y_test.ndim == 1 else y_test.iloc[:, 0]

# Initialize and fit the logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Make predictions on the test data
log_test_pred = log_model.predict(X_test)

log_train_pred = log_model.predict(X_train)
print("Metrics for test data:")
get_report(y_test, log_test_pred, 'LinReg')
print("\nMetrics for train data:")
get_report(y_train, log_train_pred)

def normalize_predictions(pred):
    new_pred = [i if i <= 5 else 5 for i in pred]
    new_pred = [i if i >= 1 else 1 for i in new_pred]
    new_pred = [round(i) for i in new_pred]
    return new_pred

tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)

tree_pred_test = normalize_predictions(tree_model.predict(X_test))
tree_pred_train = normalize_predictions(tree_model.predict(X_train))
print("Metrics for test data:")
get_report(y_test, tree_pred_test, 'DesTree')
print("\nMetrics for train data:")
get_report(y_train, tree_pred_train)
random_forest_model = RandomForestRegressor(max_features=100)
random_forest_model.fit(X_train, y_train)

forest_test_pred = normalize_predictions(random_forest_model.predict(X_test))
forest_train_pred = normalize_predictions(random_forest_model.predict(X_train))

print("Metrics for test data:")
get_report(y_test, forest_test_pred, 'RanFor')
print("\nMetrics for train data:")
get_report(y_train, forest_train_pred)
models_name = models_metrics.index.tolist()
plt.figure(figsize=(7,7))
plt.bar(models_name, models_metrics['RMSE'], label='RMSE')
plt.bar(models_name, models_metrics['Accuracy'], label='Accuracy')
plt.bar(models_name, models_metrics['R^2'], label='R^2')
plt.title('Metrics')
plt.legend()
plt.show()
tfidf = TfidfVectorizer(min_df=0.001)
tfidf_data = tfidf.fit_transform(X)

tfidf_data = pd.DataFrame(tfidf_data.toarray(), columns=tfidf.get_feature_names_out())
tfidf_data.head()
tfidf = TfidfVectorizer(min_df=0.001)
tfidf_data = tfidf.fit_transform(X)

tfidf_data = pd.DataFrame(tfidf_data.toarray(), columns=tfidf.get_feature_names_out())
tfidf_data.head()
X_train_idf = tfidf_data.loc[X_train.index,:]
X_test_idf = tfidf_data.loc[X_test.index,:]

print(X_train_idf.shape)
print(X_test_idf.shape)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_idf, y_train)

idf_log_test = log_model.predict(X_test_idf)
idf_log_train = log_model.predict(X_train_idf)
print("Metrics for test data")
get_report(y_test, idf_log_test, 'LinReg')
print("\nMetrics for train data")
get_report(y_train, idf_log_train)
tree_model = DecisionTreeRegressor(max_depth=10)
tree_model.fit(X_train_idf, y_train)

idf_tree_test = normalize_predictions(tree_model.predict(X_test_idf))
idf_tree_train = normalize_predictions(tree_model.predict(X_train_idf))

param_grid = {
    "max_depth": [3,5,10,20,50,70],
}

clf = DecisionTreeRegressor(random_state=42)
grid_cv = GridSearchCV(clf, param_grid, n_jobs=-1).fit(X_train_idf, y_train)

print("Param for GS", grid_cv.best_params_)
print("CV score for GS", grid_cv.best_score_)

r2 = list()
x = [10, 20, 30, 50, 70, 100]

for depth in x:
    tree_model = DecisionTreeRegressor(max_depth=depth)
    tree_model.fit(X_train_idf, y_train)
    idf_tree_test = normalize_predictions(tree_model.predict(X_test_idf))
    r2.append(r2_score(y_test, idf_tree_test))

tree_model = DecisionTreeRegressor(max_depth=20)
tree_model.fit(X_train_idf, y_train)

idf_tree_test = normalize_predictions(tree_model.predict(X_test_idf))
idf_tree_train = normalize_predictions(tree_model.predict(X_train_idf))
mse_test = list()
mse_train = list()
x = [20, 50, 100, 200, 300, 500]

for num in x:
    random_forest_model = RandomForestRegressor(max_features=num)
    random_forest_model.fit(X_train_idf, y_train)
    idf_forest_test = normalize_predictions(random_forest_model.predict(X_test_idf))
    idf_forest_train = normalize_predictions(random_forest_model.predict(X_train_idf))
    mse_test.append(mean_squared_error(y_test, idf_forest_test))
    mse_train.append(mean_squared_error(y_train, idf_forest_train))

random_forest_model = RandomForestRegressor(max_features=200)
random_forest_model.fit(X_train_idf, y_train)
idf_forest_test = normalize_predictions(random_forest_model.predict(X_test_idf))
idf_forest_train = normalize_predictions(random_forest_model.predict(X_train_idf))

print("Metrics for test data")
get_report(y_test, idf_forest_test, 'RanFor')
print("\nMetrics for test data")
get_report(y_train, idf_forest_train)

def predict_with_model(model, X_test_idf):
    idf_forest_test = normalize_predictions(random_forest_model.predict(X_test_idf))
    return idf_forest_test
if __name__ == "__main__":
    review = "This product exceeded my expectations. It's really amazing!"
    predicted_rating = predict_with_model(random_forest_model, review)
    print(f"Predicted Rating: {predicted_rating}")
