import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from skopt import BayesSearchCV

# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer

# Loading the Dataset
dataset = load_dataset("amazon_us_reviews", 'Video_Games_v1_00', split='train')
df = pd.DataFrame(dataset)[:1000]

X_train, X_test, y_train, y_test = train_test_split(df['review_body'], df['star_rating'], test_size=0.3)


# Defining the model
model = Pipeline([('vect', TfidfVectorizer()),
                  ('clf', RandomForestClassifier())])

# Pipeline parameters search spaces
search_spaces = {"vect__max_features": Integer(1000, 1500),
                 "vect__max_df": Real(0, 1, prior='uniform'),
                 "clf__n_estimators": Integer(200,  400)
                }

# Bayesian grid search on the pipeline
opt = BayesSearchCV(model, search_spaces, cv=3)
opt.fit(X_train, y_train)

print(opt.best_score_)
