import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# Loading the Dataset
dataset = load_dataset("amazon_us_reviews", 'Video_Games_v1_00', split='train')
df = pd.DataFrame(dataset)[:1000]

X_train, X_test, y_train, y_test = train_test_split(df['review_body'], df['star_rating'], test_size=0.3)


# Defining the model
model = Pipeline([('vect', TfidfVectorizer()),
                  ('clf', RandomForestClassifier())])

# Pipeline estimators parameters
param_grid = {"vect__max_features": [1000, 1500],
              "clf__n_estimators": [200, 300, 400],
              "clf__criterion": ["gini", "entropy"]}

# Successive halving grid search on the pipeline
halving_gs = HalvingGridSearchCV(model, param_grid=param_grid, cv=3)
halving_gs.fit(X_train, y_train)

print(halving_gs.best_score_)
