import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Loading the Dataset
dataset = load_dataset("amazon_us_reviews", 'Video_Games_v1_00', split='train')
df = pd.DataFrame(dataset)[:1000]

X_train, X_test, y_train, y_test = train_test_split(df['review_body'], df['star_rating'], test_size=0.3)


# Defining the model
model = Pipeline([('vect', TfidfVectorizer()),
                  ('clf', RandomForestClassifier())])

# Pipeline estimators parameters
distributions = {"vect__max_features": [1000, 1500],
                 "vect__max_df": uniform(),
                 "clf__n_estimators": [200, 300, 400]}

# Grid search on the pipeline
grid_search = RandomizedSearchCV(model, param_distributions=distributions, cv=3)
grid_search.fit(X_train, y_train)

print(grid_search.best_score_)
