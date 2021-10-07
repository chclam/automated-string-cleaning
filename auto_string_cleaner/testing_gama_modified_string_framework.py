from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from gama import GamaClassifier
from auto_string_cleaner import main
import openml as oml
import traceback
import pickle

METHOD = 'modified_string_framework'

if __name__ == '__main__':

    list_of_datasets = [
        # 31, 1464, 334, 50, 333, 1504, 3, 1494, 1510, 1489, 37, 1063, 1467, 1067, 1480, 1068, 1050,
        1462, 1049, 335, 42, 18, 183, 22, 54, 182, 11, 15, 469, 188, 307, 4538, 1497, 29,
        # 1466, 23, 375, 36, 6332, 40499, 38, 60, 23381, 23380, 24, 451, 470, 2, 40496, 1549
    ]

    accuracies = {}

    for dataset_id in list_of_datasets:
        data = oml.datasets.get_dataset(dataset_id)
        X, y, _, _ = data.get_data(target=data.default_target_attribute)

        try:
            X, y = main.run(data=X, y=y, dense_encoding=False)
        except Exception:
            with open(f"errors/triple_test/{METHOD}/framework-{dataset_id}.txt", "w") as log:
                traceback.print_exc(file=log)
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

        print(f"Starting `fit` on the {METHOD} dataset with id {dataset_id} which will take roughly 5 minutes.")
        try:
            automl = GamaClassifier(max_total_time=300, store="nothing", max_memory_mb=16_000_000)
            automl.fit(X_train, y_train)
            label_predictions = automl.predict(X_test)
            probability_predictions = automl.predict_proba(X_test)
            accuracy = accuracy_score(y_test, label_predictions)
            accuracies[dataset_id] = accuracy

            with open(f"results/{METHOD}/{dataset_id}.txt", "x") as f:
                print('accuracy:', accuracy, file=f)
                print('log loss:', log_loss(y_test, probability_predictions), file=f)
                # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
                print('log_loss', automl.score(X_test, y_test), file=f)
        except Exception:
            with open(f"errors/triple_test/{METHOD}/GAMA-{dataset_id}.txt", "w") as log:
                traceback.print_exc(file=log)
            continue

    with open(f"results/{METHOD}/accuracies.pickle", 'wb') as handle:
        pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)
