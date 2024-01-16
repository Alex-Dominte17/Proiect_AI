import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

url = "HH_Provider_Oct2023.csv"
dataset = pd.read_csv(url)

#Eliminam coloanele irelevante:
footnote_cols = [col for col in dataset.columns if 'Footnote' in col]
dataset = dataset.drop(footnote_cols, axis=1)

irrelevant_cols = ['CMS Certification Number (CCN)', 'Address', 'City/Town', 'ZIP Code', 'Telephone Number', 'Type of Ownership', 'Certification Date']
dataset = dataset.drop(irrelevant_cols, axis=1)

#Eliminam coloanele cu mai mult de 75% valori lipsa:
percentage_missing_values = ((dataset == '-').sum() / len(dataset)) * 100
columns_to_drop = percentage_missing_values[percentage_missing_values >= 75].index.tolist()
dataset = dataset.drop(columns_to_drop, axis=1)

dataset.replace('-', float('nan'), inplace=True)
dataset.replace("Yes", "1", inplace=True)
dataset.replace("No", "-1", inplace=True)

converted_dataset = pd.DataFrame()

#Convertim datele la tip numeric
for column in dataset.columns:
    numeric_values = []

    for value in dataset[column]:
        try:
            numeric_value = pd.to_numeric(value)
            numeric_values.append(numeric_value)
        except ValueError:
            numeric_values.append(value)

    converted_dataset[column] = numeric_values

#Interpolam valorile lipsa de pe coloanele numerice
numeric_columns = converted_dataset.select_dtypes(include='float64')
for col in numeric_columns.columns:
    converted_dataset[col] = converted_dataset[col].interpolate(method='linear', limit_direction='both')

dataset = converted_dataset

#Inlocuim valorile lipsa de pe coloanele categorice
dataset = dataset.fillna(dataset.mode().iloc[0])

# print("Coloanele rămase:")
# print(dataset.columns)

#Analiza exploratorie a setului de date
print("Minim:", dataset['Quality of patient care star rating'].min())
print("Maxim:", dataset['Quality of patient care star rating'].max())
print(dataset['Quality of patient care star rating'].value_counts())

plt.figure(figsize=(12, 8))
sns.histplot(dataset['Quality of patient care star rating'], bins=np.arange(0, 5.5, 0.5))
plt.title('Distribuția variabilei \'Quality of patient care star rating\'')
plt.xticks(np.arange(0, 5.5, 0.5))
plt.show()

plt.figure(figsize=(12, 8))
sns.kdeplot(dataset['Quality of patient care star rating'], color='blue', fill=True)
plt.title('Distribuția variabilei \'Quality of patient care star rating\'')
plt.xlabel('Quality of patient care star rating')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(dataset["How often patients' breathing improved"])
plt.title("Distribuția variabilei 'How often patients' breathing improved'")
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(dataset["How often patients got better at taking their drugs correctly by mouth"])
plt.title("Distribuția variabilei 'How often patients got better at taking their drugs correctly by mouth'")
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(dataset["How often the home health team determined whether patients received a flu shot for the current flu season"])
plt.title("Distribuția variabilei 'How often the home health team determined whether patients received a flu shot for the current flu season'")
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(dataset["How often patients got better at bathing"])
plt.title("Distribuția variabilei 'How often patients got better at bathing'")
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

for column in dataset.columns:
    try:
        print(column, "----Mean: ", dataset[column].mean(), " Median: ", dataset[column].median())
    except:
        pass

#Selectia atributelor cu SelectKBest
numeric_columns = dataset.select_dtypes(include='float64')
new_dataset = pd.DataFrame(numeric_columns)

y = new_dataset['Quality of patient care star rating']
X = new_dataset.drop('Quality of patient care star rating', axis=1)

num_features_to_select = 10
selector = SelectKBest(f_classif, k=num_features_to_select)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Caracteristici selectate:", selected_features)


#Selectia atributelor cu PCA
num_features_to_select_pca = 10
pca_selector = PCA(n_components=num_features_to_select_pca)
X_pca_selected = pca_selector.fit_transform(X)

loadings = pca_selector.components_
feature_importance = np.sum(np.abs(loadings), axis=0)
sorted_features = np.argsort(feature_importance)[::-1]
selected_features_pca = X.columns[sorted_features[:num_features_to_select_pca]]
print("Selected Features (PCA):", selected_features_pca)






