import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif

url = "dataset.csv"
dataset = pd.read_csv(url)

dataset_for_pca = copy.deepcopy(dataset)

dataset.replace("-", "0", inplace=True)
dataset.replace("Yes", "1", inplace=True)
dataset.replace("No", "-1", inplace=True)

converted_dataset = pd.DataFrame()

for column in dataset.columns:
    numeric_values = []

    for value in dataset[column]:
        try:
            numeric_value = pd.to_numeric(value)
            numeric_values.append(numeric_value)
        except ValueError:
            numeric_values.append(value)

    converted_dataset[column] = numeric_values

dataset = converted_dataset
print("Minim:", dataset['Quality of patient care star rating'].min())
print("Maxim:", dataset['Quality of patient care star rating'].max())
print(dataset['Quality of patient care star rating'].value_counts())
print("----")

plt.figure(figsize=(12, 8))
sns.histplot(dataset['Quality of patient care star rating'], bins=np.arange(0, 5.5, 0.5), kde=True)
plt.title('Distribuția variabilei \'Quality of patient care star rating\'')
plt.xticks(np.arange(0, 5.5, 0.5))  # Set ticks at 0, 0.5, 1.0, ..., 5.0
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x=dataset['Quality of patient care star rating'].value_counts().index,
            y=dataset['Quality of patient care star rating'].value_counts(),
            color='blue')
plt.title('Distribuția variabilei \'Quality of patient care star rating\'')
plt.xlabel('Quality of patient care star rating')
plt.ylabel('Count')
plt.show()

for column in dataset.columns:
    try:
        print(column, "----Mean: ", dataset[column].mean(), " Median: ", dataset[column].median())
    except:
        pass

numeric_columns = dataset.select_dtypes(include='float64')
new_dataset = pd.DataFrame(numeric_columns)

y = new_dataset['Quality of patient care star rating']
X = new_dataset.drop('Quality of patient care star rating', axis=1)

num_features_to_select = 5
selector = SelectKBest(f_classif, k=num_features_to_select)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Caracteristici selectate:", selected_features)
