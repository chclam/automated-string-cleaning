#!/usr/bin/env python3
import openml
from auto_string_cleaner import main as sc

D_ID = 10

if __name__ == "__main__":
  dataset = openml.datasets.get_dataset(D_ID)
  X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
  )
  
  print("Dataset loaded")
  schema, _ = sc.inference_ptype(X)
  print([col.type for col in schema.cols.values()])

