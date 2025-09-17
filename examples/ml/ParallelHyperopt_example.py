
from hyperopt import hp

from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.HyperOptSelector_parallel import (
    HyperOptSelector_)  # ← Make sure the import path matches your project
from astrodata.ml.models.SklearnModel import SklearnModel

import os





#todo le cancelliamo?  Sono vuote, anche il logfile

# directory for mongo workers, set it to find its files
# (if you want the training to run in parallel, otherwise you don't  need it)
path_workers_job = "C:/Users/yourfolder/PycharmProjects/yourproject/MongoDB_workers_jobs"
os.makedirs(path_workers_job, exist_ok=True)
os.chdir(path_workers_job)


# Instantiate the SklearnModel with LinearSVC and a metric

model_list = ["LinearSVC"]

model_mapping = {
    "LinearSVC": SklearnModel(model_class=LinearSVC),
    # add other models
}


accuracy = SklearnMetric(accuracy_score)
f1 = SklearnMetric(f1_score, average="micro")
metrics = [accuracy, f1] #please choose accordingly to the model, or leave it None





# set it equal to 1 to use Cross Validation, 0 to avoid it
use_cv = 0





# load your data

data = fetch_covtype()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)






# set ypur param space

param_space = {
"model": hp.choice("model", model_list),
"C": hp.choice("C",[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 150]),
"max_iter": hp.choice("max_iter",[100, 150, 200, 250, 500, 750, 1000]),
"tol": hp.choice("tol", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]),
"fit_intercept": hp.choice("fit_intercept", [True, False]),
"class_weight": hp.choice("class_weight",[None, "balanced"]),
}





# available  cores on your machine
n_cores_available = os.cpu_count()
print ("You have", n_cores_available, "cores available on this machine.")


# choose n_core_choosen = 1 if you do NOT want to parallelize on multiple cores.
# If you want to parallelize the training, please install MongoDB and set the mongo_url

n_cores_chosen = 8


# IMPORTANT: Use "mongo://" as the protocol, not "mongodb://"
# example URL format: mongo://localhost:27017/database_name.collection_name


mongo_url = "mongo://localhost:27017/hyperopt_db/jobs"







# todo nella versione finale bisogna eliminare il tratto basso in HyperOptSelector_
#  che al momento c'è per poter chiamare sia HyperOptSelector in parallelo che
#  quello del codice vecchio senza imbrogliarsi avendo due HyperOptSelector con
#  lo stesso identico nome


if use_cv == 1:
    hos = HyperOptSelector_(
        n_cores=n_cores_chosen,
        mongo_url= mongo_url,
        show_worker_terminal=False,
        # ^ False to avoid seeing n terminal windows opening suddenly,
        # | but if you have problems, they can be informative
        param_space=param_space,
        model_mapping=model_mapping,
        model_list=model_list,
        scorer=accuracy,
        use_cv=True,
        cv=5,
        random_state=42,
        max_evals=100,  # You can increase this for a more thorough search
        metrics=metrics,
    )

else:
    hos = HyperOptSelector_(
       n_cores=n_cores_chosen,
       mongo_url= mongo_url,
       show_worker_terminal=False,
       # ^ False to avoid seeing n terminal windows opening suddenly,
       # | but if you have problems, they can be informative
       param_space=param_space,
       model_mapping=model_mapping, #se no da None
       model_list=model_list, #
       scorer=accuracy,
       use_cv=False,
       cv=5,
       random_state=42,
       max_evals=100,  # You can increase this for a more thorough search
       metrics=metrics,
   )



hos.fit(X_train, y_train)

print(f"Best parameters found:", hos.get_best_params())
print(f"Best metrics:", hos.get_best_metrics())
print(f"Best metrics:", hos.get_best_model().get_params())







