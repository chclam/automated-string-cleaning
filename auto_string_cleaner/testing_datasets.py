import pandas as pd

from auto_string_cleaner import main
import openml as oml

# The dataset John used
# X = pd.read_csv('datasets/xAPI-Edu-Data.csv')
# X = main.run(data=X)

# credit: https://www.openml.org/d/31
# Check mail to Marcos for explanation why this breaks.
# credit = oml.datasets.get_dataset(31)
# X, y, _, _ = credit.get_data(target=credit.default_target_attribute)

# bank-marketing: https://www.openml.org/d/1461
# This works, it does produce a weird ignored exception but the final DataFrame is fine.
# bank_marketing = oml.datasets.get_dataset(1461)
# X, y, _, _ = bank_marketing.get_data(target=bank_marketing.default_target_attribute)

# soybean: https://www.openml.org/d/42
# Changed one line of code to not break at missing values (line 115), now breaks similarly to credit dataset.
# soybean = oml.datasets.get_dataset(42)
# X, y, _, _ = soybean.get_data(target=soybean.default_target_attribute)

# eucalyptus: https://www.openml.org/d/188
# Works after making changes to 2 lines in handle_missing.py (lines 114, 115).
# eucalyptus = oml.datasets.get_dataset(188)
# X, y, _, _ = eucalyptus.get_data(target=eucalyptus.default_target_attribute)

# sponge: https://www.openml.org/d/1001
# Many binary columns with values NO and SI. SI is seen as an outlier and replaced by NO. Very bad, but code works.
# sponge = oml.datasets.get_dataset(1001)
# X, y, _, _ = sponge.get_data(target=sponge.default_target_attribute)

# BNG(anneal,nominal,1000000): https://www.openml.org/d/70
# Weird outliers detected, but code does work on values like "B1OF3". Taking only 1% of the data to speed things up.
# bng = oml.datasets.get_dataset(70)
# X, y, _, _ = bng.get_data(target=bng.default_target_attribute)
# from sklearn.model_selection import train_test_split
# X, _, y, _ = train_test_split(X, y, train_size=0.01, random_state=1)

# ipums_la_97-small: https://www.openml.org/d/382
# Code works on this dataset while it contains many string columns and missing data. Weird outliers detected however.
# ipums_la_97 = oml.datasets.get_dataset(382)
# X, y, _, _ = ipums_la_97.get_data(target=ipums_la_97.default_target_attribute)

# ipums_la_98-small: https://www.openml.org/d/381
# Code works on this dataset while it contains many string columns and missing data. Weird outliers detected however.
# ipums_la_98 = oml.datasets.get_dataset(381)
# X, y, _, _ = ipums_la_98.get_data(target=ipums_la_98.default_target_attribute)

# ipums_la_99-small: https://www.openml.org/d/378
# Code works on this dataset while it contains many string columns and missing data. Weird outliers detected however.
# ipums_la_99 = oml.datasets.get_dataset(378)
# X, y, _, _ = ipums_la_99.get_data(target=ipums_la_99.default_target_attribute)

# blogger: https://www.openml.org/d/1463
# Simple dataset with a few categorical text columns, code works.
# blogger = oml.datasets.get_dataset(1463)
# X, y, _, _ = blogger.get_data(target=blogger.default_target_attribute)

# usp05-ft: https://www.openml.org/d/1057
# Code works on this dataset while it contains a variety of string columns and missing data. Weird outliers detected.
# usp05_ft = oml.datasets.get_dataset(1057)
# X, y, _, _ = usp05_ft.get_data(target=usp05_ft.default_target_attribute)

X, y = main.run(data=X, y=y)
