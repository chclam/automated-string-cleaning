#!/usr/bin/env python3
import openml
from auto_string_cleaner import main as sc
from ptype.Ptype import Ptype
from modules.pfsms import create_pfsm
import traceback

class TypeInferer:
  def __init__(self):
    self.ptype = self.__add_pfsms(Ptype())

  def infer(self, X):
    return self.ptype.schema_fit(X)

  def __add_pfsms(self, ptype):
    ptype.machines.forType['string'].initialize(reg_exp="[@#]*[a-zA-Z0-9 .,\\\-_%:;&]+ ?")
    names = ['coordinate', 'day', 'email', 'filepath', 'month', 'numerical', 'sentence', 'url', 'zipcode']
    machines = [
      create_pfsm.Coordinate(), create_pfsm.Day(), create_pfsm.Email(),
      create_pfsm.Filepath(), create_pfsm.Month(), create_pfsm.Numerical(),
      create_pfsm.Sentence(), create_pfsm.URL(), create_pfsm.Zipcode(),
    ]
    for name, machine in zip(names, machines):
      ptype.types.append(name)
      ptype.machines.forType[name] = machine
    return ptype

def log_error(d_id):
  with open(f"errors/{d_id}.txt", "w") as f:
    traceback.print_exc(file=f)

if __name__ == "__main__":
  with open("datasets/openml_ids.txt", "r") as f:
    D_IDS = [int(x) for x in f.read().split(", ")]

  counter = {"success": 0, "fail": 0}
  ti = TypeInferer()

  for d_id in D_IDS:
    try:
      print(f">> Loading dataset #{d_id}...")
      dataset = openml.datasets.get_dataset(d_id)
      X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
      )
    except Exception:
      log_error(d_id)
      counter["fail"] += 1
      continue

    print(f">> Infering type for #{d_id}...")
    try:
      schema = ti.infer(X)
      types = [col.type for col in schema.cols.values()]
      for t in types:
        if t not in counter:
          counter[str(t)] = 1
        else:
          counter[t] += 1
        counter["success"] += 1
    except Exception as e:
      log_error(d_id)
      counter["fail"] += 1

  with open("results/type_count.txt", "w") as f:
    f.write(str(counter))

