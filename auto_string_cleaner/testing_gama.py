from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from gama import GamaClassifier
from auto_string_cleaner import main
import openml as oml
import traceback
import pandas as pd

if __name__ == '__main__':

    list_of_datasets = [31, 42, 1461, 378]
    for dataset_id in list_of_datasets:
        data = oml.datasets.get_dataset(dataset_id)
        X, y, _, _ = data.get_data(target=data.default_target_attribute)
        X_p, y_p = main.run(data=X, y=y, dense_encoding=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
        X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_p, y_p, stratify=y_p, random_state=0)
        methods = ['unprocessed', 'preprocessed']
        for method in methods:
            print(f"Starting `fit` on the {method} dataset with id {dataset_id} which will take roughly 3 minutes.")
            if method == 'unprocessed':
                try:
                    automl = GamaClassifier(max_total_time=180, store="nothing")
                    automl.fit(X_train, y_train)
                    label_predictions = automl.predict(X_test)
                    probability_predictions = automl.predict_proba(X_test)

                    print('accuracy:', accuracy_score(y_test, label_predictions))
                    print('log loss:', log_loss(y_test, probability_predictions))
                    # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
                    print('log_loss', automl.score(X_test, y_test))
                except Exception:
                    with open(f"errors/GAMA/{method}/{dataset_id}.txt", "w") as log:
                        traceback.print_exc(file=log)
                    continue
            elif method == 'preprocessed':
                try:
                    automl = GamaClassifier(max_total_time=180, store="nothing")
                    automl.fit(X_p_train, y_p_train)
                    label_predictions = automl.predict(X_p_test)
                    probability_predictions = automl.predict_proba(X_p_test)

                    print('accuracy:', accuracy_score(y_p_test, label_predictions))
                    print('log loss:', log_loss(y_p_test, probability_predictions))
                    # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
                    print('log_loss', automl.score(X_p_test, y_p_test))
                except Exception:
                    with open(f"errors/GAMA/{method}/{dataset_id}.txt", "w") as log:
                        traceback.print_exc(file=log)
                    continue
