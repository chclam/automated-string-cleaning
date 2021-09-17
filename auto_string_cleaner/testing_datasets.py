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

import traceback
IDS = [
        490, 1072, 23380, 1024, 1093, 1037, 205, 473, 566, 567, 568, 569, 570, 575, 576, 577, 578, 510, 524, 516, 315,
        327, 328, 42673, 42878, 42880, 42882, 42883, 42768, 42897, 42898, 42804, 42805, 42986, 42987, 43003, 43004,
        43005, 43006, 43009, 42960, 42664, 42723, 42730, 42825, 42781, 41705, 42125, 41021, 41002, 41003, 41006, 41981,
        41980, 41968, 42532, 42460, 42530, 42563, 42637, 42638, 42647, 42652, 42654, 42655, 42621, 42622, 42623,
        42624, 42625, 42626, 42603, 42604, 42605, 42606, 42607, 42609, 42610, 42611, 42612, 42613, 42614, 42615, 42616,
        42617, 42618, 42619, 42620, 41091, 42172, 42177, 42133, 42107, 42167, 42169, 42123, 42159, 41430, 42359,
        41533, 42195, 42196, 42260, 42371, 43025, 43023, 43028, 43029, 456, 471, 491, 532, 539, 506,
        498, 502, 42918, 43035, 43033, 42931, 42910, 42911, 42912, 42972, 43038, 42969, 42965, 42964, 42967, 42968,
        42966, 42989, 42970, 42971, 42585, 42165, 40945, 222, 224, 204, 231, 194, 200, 232, 196, 42727, 40966, 6332,
        23381, 43085, 213, 40536, 41190, 40728, 42907, 2, 5, 7, 9, 4, 13, 15, 24, 34, 188, 27, 186, 185, 29, 49, 42, 38,
        35, 55, 52, 56, 25, 51, 57, 171, 172, 163, 342, 340, 460, 454, 474, 470, 443, 452, 449, 453, 451, 466, 382, 378,
        381, 739, 481, 738, 488, 760, 810, 757, 798, 802, 786, 831, 840, 839, 854, 842, 852, 861, 858, 844, 897, 899,
        898, 930, 940, 944, 939, 960, 966, 961, 957, 967, 968, 963, 982, 986, 972, 984, 989, 1002, 993, 1001, 992, 999,
        998, 975, 985, 990, 1010, 1000, 1003, 1008, 1018, 1007, 1023, 1017, 1057, 1101, 1102, 1109, 455
]

for dataset_id in IDS:
    try:
        print(f"Starting a new dataset with id {dataset_id}")
        data = oml.datasets.get_dataset(dataset_id)
        X, y, _, _ = data.get_data(target=data.default_target_attribute)
    except Exception:
        with open(f"errors/before_string_handling/{dataset_id}.txt", "w") as log:
            traceback.print_exc(file=log)
            continue
    try:
        X, y = main.run(data=X, y=y)
    except TypeError:
        with open(f"errors/during_string_handling/TypeError/{dataset_id}.txt", "w") as log:
            traceback.print_exc(file=log)
        continue
    except ValueError:  # ValueError could be due to there not being a target in the dataset
        try:
            X = main.run(data=X)
        except TypeError:
            with open(f"errors/during_string_handling/TypeError/{dataset_id}.txt", "w") as log:
                traceback.print_exc(file=log)
            continue
        except ValueError:
            with open(f"errors/during_string_handling/ValueError/{dataset_id}.txt", "w") as log:
                traceback.print_exc(file=log)
            continue
        except KeyError:
            with open(f"errors/during_string_handling/KeyError/{dataset_id}.txt", "w") as log:
                traceback.print_exc(file=log)
            continue
        except Exception:
            with open(f"errors/during_string_handling/Other/{dataset_id}.txt", "w") as log:
                traceback.print_exc(file=log)
            continue
        with open(f"errors/during_string_handling/ValueError/{dataset_id}.txt", "w") as log:
            traceback.print_exc(file=log)
        continue
    except KeyError:
        with open(f"errors/during_string_handling/KeyError/{dataset_id}.txt", "w") as log:
            traceback.print_exc(file=log)
        continue
    except Exception:
        with open(f"errors/during_string_handling/Other/{dataset_id}.txt", "w") as log:
            traceback.print_exc(file=log)
        continue
