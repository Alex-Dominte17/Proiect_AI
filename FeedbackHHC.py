import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
os.environ['PDOC_ALLOW_EXEC'] = '1'


url = "dataset.csv"
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


numeric_columns = dataset.select_dtypes(include='float64')
new_dataset = pd.DataFrame(numeric_columns)

dataset_for_neural_network = copy.deepcopy(new_dataset)


def display_provider_info(data, provider_name):
    """
    This function returns all the available information regarding the specified provider.

    Parameters
    ----------
        data(DataFrame): Contains the entire dataset.
        provider_name(str): Name of the wanted provider.

    Returns
    -------
        All found information about the provider/ an adequate message if nothing is found.

    """
    provider_info = data[data['Provider Name'] == provider_name]

    if not provider_info.empty:
        print("Provider Information:")
        for index, row in provider_info.iterrows():
            for col_name, value in row.items():
                print(f"{col_name}: {value}")
            print("-" * 30)
    else:
        print(f"No information found for the provider '{provider_name}'.")


def display_rating_for_provider(data, provider_name):
    """
        This function looks for the specified provider and returns their rating .

        Parameters
        ----------
            data(DataFrame): Contains the entire dataset.
            provider_name(str): Name of the wanted provider.

        Returns
        -------
            Rating of the provider/ an adequate message if nothing is found.

        """

    provider_info = data[data['Provider Name'] == provider_name]

    if not provider_info.empty:
        rating_value = provider_info['Quality of patient care star rating'].values[0]
        print(f"Quality of patient care star rating for {provider_name}: {rating_value}")
    else:
        print(f"No information found for the provider '{provider_name}'.")


def calculate_mean_and_median_for_column(data, column_name):
    """
        This function computes the mean and median for a specified column from the dataset.

        Parameters
        ----------
            data(DataFrame): Contains the entire dataset.
            column_name(str): Name of the column.

        Returns
        -------
            Column name followed by the mean and median, or an error.

        Raises
        ------
            KeyError: If the column doesn't exist


        """

    try:
        mean_value = data[column_name].mean()
        median_value = data[column_name].median()
        print(f"{column_name} ---- Mean: {mean_value}, Median: {median_value}")
    except KeyError as e:
        print(f"Error: {e}. Make sure the column name is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_mean_and_median_for_all_columns(dataset):
    """
        This function calculates the mean and the median for all found columns in the dataset.

        Parameters
        ----------
            dataset(DataFrame): Contains the entire dataset.

        Returns
        -------
            The mean and median for all columns memorizing float numbers.

        """

    for column in dataset.columns:
        try:
            print(column, "----Mean: ", dataset[column].mean(), " Median: ", dataset[column].median())
        except:
            pass


def plot_attribute_histogram(column_name, data):
    """
    This function displays the histogram of a given column in report with the target. In addition, columns of yes/no
    type are given numerical values.

    Parameters
    ----------
        dataset(DataFrame): Contains the entire dataset.
        column_name(str): Name of the columns

    Returns
    -------
        A histogram of the selected attribute(X(rating)/Y(Count)) .

    """

    # Verifica daca atributul are doar valori -1 și 1 (yes/no)
    if set(data[column_name].dropna().unique()) == {-1.0, 1.0}:
        # Inlocuieste -1 cu "No" și 1 cu "Yes"
        data[column_name].replace(-1.0, "No", inplace=True)
        data[column_name].replace(1.0, "Yes", inplace=True)

        # Creeaza histograma
        plt.figure(figsize=(12, 8))
        sns.histplot(data[column_name])
        plt.title(f'Distribution of variable \'{column_name}\'')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()

        # Afiseaza primele 10 valori
        print(f"\nSample values for '{column_name}':")
        print(data[column_name].head(20))
    else:

        plt.figure(figsize=(12, 8))
        sns.histplot(data[column_name])
        plt.title(f'Distribution of variable \'{column_name}\'')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()

        # Afișează primele 10 valori
        print(f"\nSample values for '{column_name}':")
        print(data[column_name].head(20))


def select_k_best_features(data, target_column, num_features_to_select):
    """
        This function selects a certain number of best features according to a target column.

        Parameters
        ----------
            data(DataFrame): Contains the entire dataset.
            target_column(str): Name of the column taken in consideration when selecting the features
            num_features_to_select(int): Number of features to be selected

        Returns
        -------
            A list of k-best features using SelectKBest.

        """

    numeric_columns = data.select_dtypes(include='float64')
    new_dataset = pd.DataFrame(numeric_columns)

    dataset_for_neural_network = copy.deepcopy(new_dataset)

    y = new_dataset[target_column]
    X = new_dataset.drop(target_column, axis=1)

    selector = SelectKBest(f_classif, k=num_features_to_select)
    X_selected = selector.fit_transform(X, y)

    selected_features = X.columns[selector.get_support()]

    print("Caracteristici selectate:", selected_features)

    return selected_features


def select_features_with_pca(data, target_column, num_pca_components):
    """
    This function selects a certain number of best features according to a target column.

    Parameters
    ----------
        data(DataFrame): Contains the entire dataset.
        target_column(str): Name of the column taken in consideration when selecting the features
        num_pca_components(int): Number of features to be selected

    Returns
    -------
        A list of k-best features using PCA.

    """

    numeric_columns = data.select_dtypes(include='float64')
    new_dataset = pd.DataFrame(numeric_columns)

    dataset_for_neural_network = copy.deepcopy(new_dataset)

    y = new_dataset[target_column]
    X = new_dataset.drop(target_column, axis=1)

    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    pca = PCA(n_components=num_pca_components)
    X_pca = pca.fit_transform(X_standardized)

    loadings = pca.components_
    feature_importance = np.sum(np.abs(loadings), axis=0)
    sorted_features = np.argsort(feature_importance)[::-1]
    selected_features_pca = X.columns[sorted_features[:num_pca_components]]

    print("Selected Features (PCA):", selected_features_pca)


##############################################################################################


def preprocess_data(dataset, target_column, bins, labels):
    """
    This function has the purpose to split the dataset into target column and rest of the features. It also assigns each interval of the rating a label.

    Parameters
    ----------
        dataset(DataFrame): Contains the entire dataset.
        target_column(str): Name of the target variable column
        bins(list): List of floats describing the intervals for the ratings
        labels(list): List of integers where each number corresponds to an interval from the bins

    Returns
    -------
        Target variable column and rest of features.

    """

    # Convertirea variabilei tinta in clase discrete
    dataset[target_column] = pd.cut(dataset[target_column], bins=bins, labels=labels, include_lowest=True)

    y = dataset[target_column]   #variabila tinta
    X = dataset.drop(target_column, axis=1)    #caracteristicile din setul de date

    return X, y


def train_test_random_forest(X, y, test_size=0.2, random_state=42):
    """
    This function trains and tests the random forest algorithm.

    Parameters
    ----------
        X(list): Contains all features except for the target
        y(list): The values for the target variable
        test_size(float): A number used by the algorithm to determine how the dataset will be split in train/test.
        random_state(int): Used to set the random seed for reproducibility of the results
    Returns
    -------
        A classification report (f1_score, accuracy, etc.) for the train and test sets.
    -------

    Random Forest Classification Report (Test):

                    precision    recall  f1-score   support

            1           0.82      0.76      0.79       164
            2           0.80      0.80      0.80       581
            3           0.79      0.85      0.82       842
            4           0.87      0.84      0.86       672
            5           0.92      0.64      0.75        89

        accuracy                            0.82      2348
        macro avg       0.84      0.78      0.80      2348
        weighted avg    0.82      0.82      0.82      2348


    Random Forest Classification Report (Train):

                    precision    recall  f1-score   support

            1           1.00      1.00      1.00       692
            2           1.00      1.00      1.00      2252
            3           1.00      1.00      1.00      3454
            4           1.00      1.00      1.00      2591
            5           1.00      1.00      1.00       402

        accuracy                            1.00      9391
        macro avg       1.00      1.00      1.00      9391
        weighted avg    1.00      1.00      1.00      9391
    """
    # Impartirea setului de date in set de antrenare si set de testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Antrenarea Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_classifier.fit(X_train, y_train)

    # Testarea Random Forest pe setul de testare
    rf_predictions_test = rf_classifier.predict(X_test)
    print("Random Forest Classification Report (Test):")
    print(classification_report(y_test, rf_predictions_test))
    print()

    # Testarea Random Forest pe setul de antrenare
    rf_predictions_train = rf_classifier.predict(X_train)
    print("Random Forest Classification Report (Train):")
    print(classification_report(y_train, rf_predictions_train))
    print()

    confusion_rf_test = confusion_matrix(y_test, rf_predictions_test)
    confusion_rf_train = confusion_matrix(y_train, rf_predictions_train)

    print("Random Forest Confusion Matrix (Test):")
    print(confusion_rf_test)
    print()

    print("Random Forest Confusion Matrix (Train):")
    print(confusion_rf_train)
    print()

    return rf_classifier


def train_test_svm(X, y, test_size=0.2, random_state=42):
    """
    This function trains and tests the SVM (Support Vector Machines) algorithm.

    Parameters
    ----------
        X(list): Contains all features except for the target
        y(list): The values for the target variable
        test_size(float): A number used by the algorithm to determine how the dataset will be split in train/test.
        random_state(int): Used to set the random seed for reproducibility of the results
    Returns
    -------
        A classification report (f1_score, accuracy, etc.) for the train and test sets.
    -------

    SVM Classification Report (Test):

                    precision    recall  f1-score   support

            1           0.78      0.74      0.76       164
            2           0.77      0.75      0.76       581
            3           0.77      0.83      0.80       842
            4           0.88      0.82      0.85       672
            5           0.77      0.81      0.79        89

        accuracy                            0.80      2348
        macro avg       0.79      0.79      0.79      2348
        weighted avg    0.80      0.80      0.80      2348


    SVM Classification Report (Train):

                    precision    recall  f1-score   support

            1           0.76      0.68      0.72       692
            2           0.77      0.76      0.76      2252
            3           0.82      0.84      0.83      3454
            4           0.87      0.87      0.87      2591
            5           0.84      0.77      0.81       402

        accuracy                            0.82      9391
        macro avg       0.81      0.79      0.80      9391
        weighted avg    0.82      0.82      0.82      9391

    """

    # Impartirea setului de date in set de antrenare si set de testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Antrenarea Support Vector Machine (SVM)
    svm_classifier = SVC(kernel='linear', C=1, decision_function_shape='ovo', random_state=random_state)
    svm_classifier.fit(X_train, y_train)

    # Testarea SVM pe setul de testare
    svm_predictions_test = svm_classifier.predict(X_test)
    print("SVM Classification Report (Test):")
    print(classification_report(y_test, svm_predictions_test))
    print()

    # Testarea SVM pe setul de antrenare
    svm_predictions_train = svm_classifier.predict(X_train)
    print("SVM Classification Report (Train):")
    print(classification_report(y_train, svm_predictions_train))
    print()

    confusion_svm_test = confusion_matrix(y_test, svm_predictions_test)
    confusion_svm_train = confusion_matrix(y_train, svm_predictions_train)

    print("SVM Confusion Matrix (Test):")
    print(confusion_svm_test)
    print()

    print("SVM Confusion Matrix (Train):")
    print(confusion_svm_train)
    print()

    return svm_classifier


def train_test_neural_network(X, y, test_size=0.03, random_state=47):
    """
    This function trains and tests the MLP (Multilayer Perceptron) neural network.

    Parameters
    ----------
        X(list): Contains all features except for the target
        y(list): The values for the target variable
        test_size(float): A number used by the algorithm to determine how the dataset will be split in train/test.
        random_state(int): Used to set the random seed for reproducibility of the results
    Returns
    -------
        A classification report (f1_score, accuracy, etc.) for the train and test sets.
    -------

    MLPClassifier Classification Report (Test):

                    precision    recall  f1-score   support

            1           0.87      0.87      0.87        30
            2           0.83      0.89      0.86        79
            3           0.89      0.90      0.90       129
            4           0.94      0.87      0.90        98
            5           0.79      0.88      0.83        17

        accuracy                            0.88       353
        macro avg       0.87      0.88      0.87       353
        weighted avg    0.89      0.88      0.88       353


    MLPClassifier Classification Report (Train):

                    precision    recall  f1-score   support

            1           0.84      0.90      0.87       826
            2           0.89      0.88      0.89      2754
            3           0.90      0.93      0.92      4167
            4           0.96      0.90      0.93      3165
            5           0.86      0.82      0.84       474

        accuracy                            0.91     11386
        macro avg       0.89      0.89      0.89     11386
        weighted avg    0.91      0.91      0.91     11386


    """
    # Impartirea setului de date in set de antrenare si set de testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalizarea datelor
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Antrenarea MLPClassifier
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(150,105,70,35,15),
        max_iter=2000,
        activation='relu',
        random_state=random_state,
        solver='adam',
        learning_rate='adaptive',
        batch_size=16,
        alpha=0.001,
        early_stopping=True,
        n_iter_no_change=9

        # init='uniform'
    )
    # cross_val_accuracy = cross_val_score(mlp_classifier, X_train, y_train, cv=10, scoring='accuracy')
    # print(f"Cross-Validation Accuracy: {cross_val_accuracy.mean():.2f} (+/- {cross_val_accuracy.std() * 2:.2f})")

    mlp_classifier.fit(X_train, y_train)
    # Testarea MLPClassifier pe setul de testare
    mlp_predictions_test = mlp_classifier.predict(X_test)
    print("MLPClassifier Classification Report (Test):")
    print(classification_report(y_test, mlp_predictions_test))
    print()

    # Testarea MLPClassifier pe setul de antrenare
    mlp_predictions_train = mlp_classifier.predict(X_train)
    print("MLPClassifier Classification Report (Train):")
    print(classification_report(y_train, mlp_predictions_train))
    print()

    # Matricile de confuzie
    confusion_mlp_test = confusion_matrix(y_test, mlp_predictions_test)
    confusion_mlp_train = confusion_matrix(y_train, mlp_predictions_train)

    print("MLPClassifier Confusion Matrix (Test):")
    print(confusion_mlp_test)
    print()

    print("MLPClassifier Confusion Matrix (Train):")
    print(confusion_mlp_train)
    print()

    return mlp_classifier


def train_test_logistic_regression(X, y, test_size=0.2, random_state=42):
    """
    This function trains and tests the logistic regression algorithm.

    Parameters
    ----------
        X(list): Contains all features except for the target
        y(list): The values for the target variable
        test_size(float): A number used by the algorithm to determine how the dataset will be split in train/test.
        random_state(int): Used to set the random seed for reproducibility of the results
    Returns
    -------
        A classification report (f1_score, accuracy, etc.) for the train and test sets.
    -------

    LogisticRegression Classification Report (Test):

                    precision    recall  f1-score   support

            1           0.79      0.70      0.74       164
            2           0.77      0.73      0.75       581
            3           0.76      0.83      0.79       842
            4           0.86      0.82      0.84       672
            5           0.77      0.75      0.76        89

        accuracy                            0.79      2348
        macro avg       0.79      0.77      0.78      2348
        weighted avg    0.79      0.79      0.79      2348


    LogisticRegression Classification Report (Train):

                    precision    recall  f1-score   support

            1           0.78      0.65      0.71       692
            2           0.76      0.74      0.75      2252
            3           0.80      0.84      0.82      3454
            4           0.85      0.88      0.86      2591
            5           0.86      0.73      0.79       402

        accuracy                            0.81      9391
        macro avg       0.81      0.77      0.79      9391
        weighted avg    0.80      0.81      0.80      9391

    """


    # Impartirea setului de date in set de antrenare si set de testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalizarea datelor
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Antrenarea LogisticRegression
    logistic_regression_classifier = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=random_state, multi_class="multinomial")
    logistic_regression_classifier.fit(X_train, y_train)

    # Testarea LogisticRegression pe setul de testare
    logistic_regression_predictions_test = logistic_regression_classifier.predict(X_test)
    print("LogisticRegression Classification Report (Test):")
    print(classification_report(y_test, logistic_regression_predictions_test))
    print()

    # Testarea LogisticRegression pe setul de antrenare
    logistic_regression_predictions_train = logistic_regression_classifier.predict(X_train)
    print("LogisticRegression Classification Report (Train):")
    print(classification_report(y_train, logistic_regression_predictions_train))
    print()

    # Matricile de confuzie
    confusion_logistic_regression_test = confusion_matrix(y_test, logistic_regression_predictions_test)
    confusion_logistic_regression_train = confusion_matrix(y_train, logistic_regression_predictions_train)

    print("LogisticRegression Confusion Matrix (Test):")
    print(confusion_logistic_regression_test)
    print()

    print("LogisticRegression Confusion Matrix (Train):")
    print(confusion_logistic_regression_train)
    print()

    return logistic_regression_classifier


def plot_roc_curves_rf_and_svm(X, y, test_size=0.2, random_state=42):
    """
    This function trains and tests the random forest and SVM algorithms and compares them using ROC (Receiver Operating Characteristic) curves.

    Parameters
    ----------
        X(list): Contains all features except for the target
        y(list): The values for the target variable
        test_size(float): A number used by the algorithm to determine how the dataset will be split in train/test.
        random_state(int): Used to set the random seed for reproducibility of the results
    Returns
    -------
        A graphical representation comparing the true positives and false positives cases of the two algorithms.

    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Antrenarea si testarea Random Forest
    rf_classifier = train_test_random_forest(X, y, test_size, random_state)
    rf_probabilities = rf_classifier.predict_proba(X_test)

    # Calcularea curbelor ROC pentru Random Forest
    rf_fpr = dict()
    rf_tpr = dict()
    rf_auc = dict()
    for i in range(len(np.unique(y_test))):
        rf_fpr[i], rf_tpr[i], _ = roc_curve(label_binarize(y_test, classes=np.unique(y_test))[:, i],
                                            rf_probabilities[:, i])
        rf_auc[i] = auc(rf_fpr[i], rf_tpr[i])

    # Antrenarea si testarea SVM
    svm_classifier = train_test_svm(X, y, test_size, random_state)
    svm_probabilities = svm_classifier.decision_function(X_test)

    # Calcularea curbelor ROC pentru SVM
    svm_fpr = dict()
    svm_tpr = dict()
    svm_auc = dict()
    for i in range(len(np.unique(y_test))):
        svm_fpr[i], svm_tpr[i], _ = roc_curve(label_binarize(y_test, classes=np.unique(y_test))[:, i],
                                              svm_probabilities[:, i])
        svm_auc[i] = auc(svm_fpr[i], svm_tpr[i])

    # Afisarea curbelor ROC intr-un singur grafic
    plt.figure(figsize=(8, 6))

    for i in range(len(np.unique(y_test))):
        plt.plot(rf_fpr[i], rf_tpr[i], lw=2, label='Random Forest {} (AUC = {:.2f})'.format(i, rf_auc[i]))
    for i in range(len(np.unique(y_test))):
        plt.plot(svm_fpr[i], svm_tpr[i], lw=2, label='SVM Class {} (AUC = {:.2f})'.format(i, svm_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('Rata de fals pozitiv (FPR)')
    plt.ylabel('Rata de adevărat pozitiv (TPR)')
    plt.title('Curbe ROC pentru Random Forest și SVM (One-Vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_curves_rf_and_nn(X, y, test_size=0.2, random_state=42):
    """
    This function trains and tests the random forest and MLP algorithms and compares them using ROC (Receiver Operating Characteristic) curves.

    Parameters
    ----------
        X(list): Contains all features except for the target
        y(list): The values for the target variable
        test_size(float): A number used by the algorithm to determine how the dataset will be split in train/test.
        random_state(int): Used to set the random seed for reproducibility of the results
    Returns
    -------
        A graphical representation comparing the true positives and false positives cases of the two algorithms.

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Antrenarea si testarea Random Forest
    rf_classifier = train_test_random_forest(X, y, test_size, random_state)
    rf_probabilities = rf_classifier.predict_proba(X_test)

    # Calcularea curbelor ROC pentru Random Forest
    rf_fpr = dict()
    rf_tpr = dict()
    rf_auc = dict()
    for i in range(len(np.unique(y_test))):
        rf_fpr[i], rf_tpr[i], _ = roc_curve(label_binarize(y_test, classes=np.unique(y_test))[:, i],
                                            rf_probabilities[:, i])
        rf_auc[i] = auc(rf_fpr[i], rf_tpr[i])

    # Antrenarea si testarea Neural Network
    nn_classifier = train_test_neural_network(X, y, test_size, random_state)
    nn_probabilities = nn_classifier.predict_proba(X_test)

    # Calcularea curbelor ROC pentru Neural Network
    nn_fpr = dict()
    nn_tpr = dict()
    nn_auc = dict()
    for i in range(len(np.unique(y_test))):
        nn_fpr[i], nn_tpr[i], _ = roc_curve(label_binarize(y_test, classes=np.unique(y_test))[:, i],
                                            nn_probabilities[:, i])
        nn_auc[i] = auc(nn_fpr[i], nn_tpr[i])

    # Afisarea curbelor ROC intr-un singur grafic
    plt.figure(figsize=(8, 6))

    for i in range(len(np.unique(y_test))):
        plt.plot(rf_fpr[i], rf_tpr[i], lw=2, label='Random Forest {} (AUC = {:.2f})'.format(i, rf_auc[i]))
    for i in range(len(np.unique(y_test))):
        plt.plot(nn_fpr[i], nn_tpr[i], lw=2, label='Neural Network {} (AUC = {:.2f})'.format(i, nn_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('Rata de fals pozitiv (FPR)')
    plt.ylabel('Rata de adevărat pozitiv (TPR)')
    plt.title('Curbe ROC pentru Random Forest și Neural Network (One-Vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_curves_svm_and_nn(X, y, test_size=0.2, random_state=42):
    """
    This function trains and tests the MLP and SVM algorithms and compares them using ROC (Receiver Operating Characteristic) curves.

    Parameters
    ----------
        X(list): Contains all features except for the target
        y(list): The values for the target variable
        test_size(float): A number used by the algorithm to determine how the dataset will be split in train/test.
        random_state(int): Used to set the random seed for reproducibility of the results
    Returns
    -------
        A graphical representation comparing the true positives and false positives cases of the two algorithms.

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Antrenarea si testarea SVM
    svm_classifier = train_test_svm(X, y, test_size, random_state)
    svm_probabilities = svm_classifier.decision_function(X_test)

    # Calcularea curbelor ROC pentru SVM
    svm_fpr = dict()
    svm_tpr = dict()
    svm_auc = dict()
    for i in range(len(np.unique(y_test))):
        svm_fpr[i], svm_tpr[i], _ = roc_curve(label_binarize(y_test, classes=np.unique(y_test))[:, i],
                                              svm_probabilities[:, i])
        svm_auc[i] = auc(svm_fpr[i], svm_tpr[i])

    # Antrenarea si testarea Neural Network
    nn_classifier = train_test_neural_network(X, y, test_size, random_state)
    nn_probabilities = nn_classifier.predict_proba(X_test)

    # Calcularea curbelor ROC pentru Neural Network
    nn_fpr = dict()
    nn_tpr = dict()
    nn_auc = dict()
    for i in range(len(np.unique(y_test))):
        nn_fpr[i], nn_tpr[i], _ = roc_curve(label_binarize(y_test, classes=np.unique(y_test))[:, i],
                                            nn_probabilities[:, i])
        nn_auc[i] = auc(nn_fpr[i], nn_tpr[i])

    # Afisarea curbelor ROC intr-un singur grafic
    plt.figure(figsize=(12, 8))

    for i in range(len(np.unique(y_test))):
        plt.plot(svm_fpr[i], svm_tpr[i], lw=2, label='SVM {} (AUC = {:.2f})'.format(i, svm_auc[i]))
    for i in range(len(np.unique(y_test))):
        plt.plot(nn_fpr[i], nn_tpr[i], lw=2, label='Neural Network {} (AUC = {:.2f})'.format(i, nn_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('Rata de fals pozitiv (FPR)')
    plt.ylabel('Rata de adevărat pozitiv (TPR)')
    plt.title('Curbe ROC pentru SVM și Neural Network (One-Vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_curve_logistic_regression(X, y, test_size=0.2, random_state=42):
    """
    This function trains and tests the logistic regression algorithm and displays its afferent ROC (Receiver Operating Characteristic) curve.

    Parameters
    ----------
        X(list): Contains all features except for the target
        y(list): The values for the target variable
        test_size(float): A number used by the algorithm to determine how the dataset will be split in train/test.
        random_state(int): Used to set the random seed for reproducibility of the results
    Returns
    -------
        A graphical representation of the true positives and false positives cases of the algorithm.
    -------

    ![Alt text](images/Regression.png)
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Training Logistic Regression
    logistic_regression_classifier = train_test_logistic_regression(X, y, test_size, random_state)

    # Binarize the labels for each class separately
    y_test_bin = label_binarize(y_test, classes=np.unique(y))

    # Getting predicted probabilities for each class
    logistic_regression_probabilities = logistic_regression_classifier.predict_proba(X_test)

    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(np.unique(y))):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], logistic_regression_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotting ROC curve for each class
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y))):
        plt.plot(fpr[i], tpr[i], lw=2, label='Class {} (AUC = {:.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Logistic Regression (One-Vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()


def make_predictions(selected_features, X, rf_classifier, svm_classifier):
    """
    This function tests the predictions of the random forest and SVM using inputs for the best features and the mean for the rest.


    Parameters
    ----------
        X(list): Contains all features except for the target
        selected_features(list): A list of the best features
        rf_classifier(RM): Trained random forest model
        svm_classifier(SVM) Trained SVM model

    Returns
    -------
        An integer representing the chosen rating interval chosen by both RM and SVM individually

    """

    print("Cele 5 atribute cele mai importante pentru predicție:", selected_features)

    input_values = {}
    for column in selected_features:
        value = input("Introduceți valoarea pentru {}: ".format(column))
        input_values[column] = pd.to_numeric(value)

    input_data = pd.DataFrame([input_values])

    # Completeaza valorile lipsa cu media din setul de date original (X)
    missing_features = set(X.columns) - set(input_data.columns)
    for missing_feature in missing_features:
        input_data[missing_feature] = X[missing_feature].mean()

    test_list = {}

    for column in X.columns:
        test_list[column] = input_data[column].astype(float).values

    test_list_data = pd.DataFrame([test_list])
    rf_prediction = rf_classifier.predict(test_list_data)
    svm_prediction = svm_classifier.predict(test_list_data)

    return rf_prediction[0], svm_prediction[0]


while True:
    print("\nMenu:")
    print("1. See all features")
    print("2. See graph for a specific feature")
    print("3. See provider information")
    print("4. See provider rating")
    print("5. Calculate mean and median for a feature")
    print("6. Calculate mean and median for all features")
    print("7. Select K best features")
    print("8. Select features with PCA")
    print("9. Train and test Random Forest")
    print("10. Train and test SVM")
    print("11. Train and test Neural Network")
    print("12. Compare algorithms (ROC curves) - Random Forest and SVM")
    print("13. Compare algorithms (ROC curves) - Random Forest and Neural Networks")
    print("14. Compare algorithms (ROC curves) - SVM and Neural Networks")
    print("15. Make predictions")
    print("16. Train and test Logistic Regression")
    print("17. ROC curve for Logistic Regression")
    print("0. Exit")

    choice = input("Choose an option: ")

    if choice == '1':
        print(dataset.columns)
    elif choice == '2':
        column_name = input("Enter the name of the attribute: ")
        try:
            plot_attribute_histogram(column_name, dataset)
        except KeyError:
            print(f"The column '{column_name}' does not exist in the dataset.")
    elif choice == '3':
        provider_name = input("Enter the name of the provider: ")
        display_provider_info(dataset, provider_name)
    elif choice == '4':
        provider_name = input("Enter the name of the provider: ")
        display_rating_for_provider(dataset, provider_name)
    elif choice == '5':
        column_name = input("Enter the name of the column: ")
        calculate_mean_and_median_for_column(dataset, column_name)
    elif choice == '6':
        calculate_mean_and_median_for_all_columns(dataset)
    elif choice == '7':
        num_features_to_select = int(input("Enter the number of features to select: "))
        select_k_best_features(dataset, 'Quality of patient care star rating', num_features_to_select)
    elif choice == '8':
        num_features_to_select_pca = int(input("Enter the number of features to select: "))
        select_features_with_pca(dataset, 'Quality of patient care star rating', num_features_to_select_pca)
    elif choice == '9':
        bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        labels = [0, 1, 2, 3, 4, 5]
        X, y = preprocess_data(dataset_for_neural_network, 'Quality of patient care star rating', bins, labels)
        train_test_random_forest(X, y)
    elif choice == '10':
        bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        labels = [0, 1, 2, 3, 4, 5]
        X, y = preprocess_data(dataset_for_neural_network, 'Quality of patient care star rating', bins, labels)
        train_test_svm(X, y)
    elif choice == '11':
        bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        labels = [0, 1, 2, 3, 4, 5]
        X, y = preprocess_data(dataset_for_neural_network, 'Quality of patient care star rating', bins, labels)
        train_test_neural_network(X, y, 0.03, 47)
    elif choice == '12':
        bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        labels = [0, 1, 2, 3, 4, 5]
        X, y = preprocess_data(dataset_for_neural_network, 'Quality of patient care star rating', bins, labels)
        plot_roc_curves_rf_and_svm(X, y)
    elif choice == '13':
        bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        labels = [0, 1, 2, 3, 4, 5]
        X, y = preprocess_data(dataset_for_neural_network, 'Quality of patient care star rating', bins, labels)
        plot_roc_curves_rf_and_nn(X, y)
    elif choice == '14':
        bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        labels = [0, 1, 2, 3, 4, 5]
        X, y = preprocess_data(dataset_for_neural_network, 'Quality of patient care star rating', bins, labels)
        plot_roc_curves_svm_and_nn(X, y)
    elif choice == '15':
        bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        labels = [0, 1, 2, 3, 4, 5]
        X, y = preprocess_data(dataset_for_neural_network, 'Quality of patient care star rating', bins, labels)
        rf_classifier = train_test_random_forest(X, y)
        svm_classifier = train_test_svm(X, y)

        # Selecteaza caracteristicile pentru predicție
        selected_features = select_k_best_features(dataset, 'Quality of patient care star rating', 5)

        # Realizeaza predictia
        rf_pred, svm_pred = make_predictions(selected_features, X, rf_classifier, svm_classifier)

        print("\nPredicție Random Forest:", rf_pred)
        print("Predicție SVM:", svm_pred)
    elif choice == '16':
        bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        labels = [0, 1, 2, 3, 4, 5]
        X, y = preprocess_data(dataset_for_neural_network, 'Quality of patient care star rating', bins, labels)
        train_test_logistic_regression(X, y)
    elif choice == '17':
        bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        labels = [0, 1, 2, 3, 4, 5]
        X, y = preprocess_data(dataset_for_neural_network, 'Quality of patient care star rating', bins, labels)
        plot_roc_curve_logistic_regression(X, y)
    elif choice == '0':
        break
    else:
        print("Invalid option. Please try again")