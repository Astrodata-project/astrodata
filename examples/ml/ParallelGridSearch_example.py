from sklearn.datasets import load_iris


import os


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC



from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.GridSearchSelector_parallel import GridSearchCVSelector_parallel
from astrodata.ml.model_selection.GridSearchSelector_parallel import GridSearchSelector_parallel
from astrodata.ml.models.SklearnModel import SklearnModel


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#vediamo quanti core ho
n_cores = os.cpu_count()



model = SklearnModel(model_class=LinearSVC, penalty="l2", loss="squared_hinge")
# evitare estimator=modello(n_jobs=1), perchè parallelizziamo una volta sola
# per esempio con RandomForestClassifier() si potrebbe fare...



accuracy = SklearnMetric(accuracy_score, greater_is_better=True)


"""
gss = GridSearchCVSelector_parallel(
    model,
    param_grid={
        "C": [0.1, 1, 10],
        "max_iter": [1000, 2000],
        "tol": [1e-3, 1e-4],
    },
    n_jobs = max(1, os.cpu_count() - 1),
    scorer=accuracy,
    cv=5,
    random_state=42,
    metrics=None,
)

print(gss)


"""

gss = GridSearchSelector_parallel(
    model,
    param_grid={
        "C": [0.1, 1, 10],
        "max_iter": [1000, 2000],
        "tol": [1e-3, 1e-4],
    },
    n_jobs = max(1, os.cpu_count() - 1),
    scorer=accuracy,
    random_state=42,
    metrics=None,
)

print(gss)







gss.fit(X_train, y_train)
print(gss.get_best_params())
print(gss.get_best_model())
print(type(gss.get_best_model()))




