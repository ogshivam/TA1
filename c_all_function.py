# data_processing.py
import pandas as pd

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv("/Users/shivampratapwar/Desktop/Customer_Churn_Prediction.csv")

def preprocess_data(df):
    """Perform data preprocessing steps."""
    print(df.shape)
    df.info()
    df.describe()
    df.isnull().sum()
    # Handle missing values, remove outliers, etc.
    return df

def calculate_prediction_percentages(confusion_matrices):
    for model, matrix in confusion_matrices.items():
        total_samples = sum(matrix.values())
        correctly_predicted = matrix["TN"] + matrix["TP"]
        incorrectly_predicted = matrix["FP"] + matrix["FN"]
        correct_percentage = (correctly_predicted / total_samples) * 100
        incorrect_percentage = (incorrectly_predicted / total_samples) * 100
        print(f"{model}:")
        print("Correctly predicted:")
        print(f"- No Churn: {matrix['TN']} / ({matrix['TN']} + {matrix['FP']}) = {correct_percentage:.1f}%")
        print(f"- Churn: {matrix['TP']} / ({matrix['TP']} + {matrix['FN']}) = {correct_percentage:.1f}%")
        print("\nIncorrectly predicted:")
        print(f"- No Churn: {matrix['FP']} / ({matrix['TN']} + {matrix['FP']}) = {incorrect_percentage:.1f}%")
        print(f"- Churn: {matrix['FN']} / ({matrix['TP']} + {matrix['FN']}) = {incorrect_percentage:.1f}%")
        print()

# Usage example:
confusion_matrices = {
    "SVM": {"TN": 1137, "FP": 18, "FN": 127, "TP": 18},
    "Random Forest": {"TN": 1141, "FP": 14, "FN": 121, "TP": 14},
    "XGBoost": {"TN": 1147, "FP": 9, "FN": 116, "TP": 9},
    "Logistic Regression": {"TN": 1149, "FP": 5, "FN": 110, "TP": 5}
}

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    # Create a figure and subplot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # Normalize the confusion matrix if desired
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Set the labels and title
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add the values to the cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', fontsize=10)

    # Add a colorbar
    fig.colorbar(im, ax=ax)

    # Show the plot
    plt.show()

def generate_and_plot_confusion_matrix(model, x_test, y_test, model_name):
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, model.predict(x_test))

    # Plot the confusion matrix
    plot_confusion_matrix(cm, classes=['No Churn', 'Churn'], title=model_name + ' Confusion Matrix')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(models, model_names, x_test, y_test):
    fig = plt.figure(figsize=(15, 5))

    for i, model in enumerate(models):
        # Calculate ROC curve and AUC for the current model
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for the current model
        plt.plot(fpr, tpr, lw=1, label=model_names[i] + " (area = %0.2f)" % roc_auc)

    # Add the diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Add the labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend(loc="lower right")

    # Show the plot
    plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(models, model_names, x_test, y_test):
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))

    for i, model in enumerate(models):
        # Calculate ROC curve and AUC for the current model
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for the current model
        axes[i].plot(fpr, tpr, color='darkorange', lw=1, label=model_names[i] + " (area = %0.2f)" % roc_auc)
        axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(model_names[i])
        axes[i].legend(loc="lower right")

    # Show the plot
    plt.tight_layout()
    plt.show()


from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(model, x_test, y_test, model_name):
    # Predict the probabilities for the test data
    y_pred_prob = model.predict_proba(x_test)[:, 1]

    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Print the AUC
    print("Area under the ROC curve for", model_name + ":", roc_auc)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label=model_name + " (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(model, x_test, y_test, model_name):
    # Predict the probabilities for the test data
    y_pred_prob = model.predict_proba(x_test)[:, 1]

    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Print the AUC
    print("Area under the ROC curve for", model_name + ":", roc_auc)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label=model_name + " (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score

def logistic_regression_with_roc_auc(x_train, y_train, x_test, y_test):
    # Create a logistic regression object
    logreg = LogisticRegression()

    # Fit the logistic regression model to the training data
    logreg.fit(x_train, y_train)

    # Predict the probabilities for the test data
    y_pred_prob = logreg.predict_proba(x_test)[:, 1]

    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Print the AUC
    print("Area under the ROC curve for Logistic Regression:", roc_auc)

    # Print the accuracy for Logistic Regression
    print("Logistic Regression accuracy:", accuracy_score(y_test, logreg.predict(x_test)))

    # Plot the ROC curve for Logistic Regression
    plt.figure()
    plt.plot(fpr, tpr, color='darkred', lw=1, label='Logistic Regression (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score


    # Create an XGBClassifier object with specified parameters
clf = XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.7,
                        subsample=0.8, nthread=10, learning_rate=0.01)
    
    # Train the XGBoost Classifier model
clf.fit(X_train, y_train)
    
    # Make predictions on the test data
y_pred = clf.predict(X_test)
    
    # Evaluate the model's performance
print('Accuracy:')
print('{}'.format(accuracy_score(y_test, y_pred)))
    
print('Classification report:')
print('{}'.format(classification_report(y_test, y_pred)))
    
print('Confusion Matrix:')
print('{}'.format(confusion_matrix(y_test, y_pred)))
    
print('Cohen kappa score:')
print('{}'.format(cohen_kappa_score(y_test, y_pred)))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest Classifier model and evaluate its performance.
    
    Args:
        X_train (numpy.ndarray or pandas.DataFrame): Training data features.
        X_test (numpy.ndarray or pandas.DataFrame): Test data features.
        y_train (numpy.ndarray or pandas.Series): Training data target labels.
        y_test (numpy.ndarray or pandas.Series): Test data target labels.
        
    Returns:
        None
    """
    # Create a Random Forest Classifier object
    rfc = RandomForestClassifier()
    
    # Train the Random Forest Classifier model
    rfc.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = rfc.predict(X_test)
    
    # Evaluate the model's performance
    print('Accuracy:')
    print('{}'.format(accuracy_score(y_test, y_pred)))
    
    print('Classification report:')
    print('{}'.format(classification_report(y_test, y_pred)))
    
    print('Confusion Matrix:')
    print('{}'.format(confusion_matrix(y_test, y_pred)))
    
    print('Cohen kappa score:')
    print('{}'.format(cohen_kappa_score(y_test, y_pred)))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

def train_and_evaluate_svc(X_train, X_test, y_train, y_test):
    """
    Train a Support Vector Classifier (SVC) model and evaluate its performance.
    
    Args:
        X_train (numpy.ndarray or pandas.DataFrame): Training data features.
        X_test (numpy.ndarray or pandas.DataFrame): Test data features.
        y_train (numpy.ndarray or pandas.Series): Training data target labels.
        y_test (numpy.ndarray or pandas.Series): Test data target labels.
        
    Returns:
        None
    """
    # Create an SVC object
    svc = SVC(kernel='rbf', decision_function_shape='ovr')
    
    # Train the SVC model
    svc.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = svc.predict(X_test)
    
    # Evaluate the model's performance
    print('Accuracy:')
    print('{}'.format(accuracy_score(y_test, y_pred)))
    
    print('Classification report:')
    print('{}'.format(classification_report(y_test, y_pred)))
    
    print('Confusion Matrix:')
    print('{}'.format(confusion_matrix(y_test, y_pred)))
    
    print('Cohen kappa score:')
    print('{}'.format(cohen_kappa_score(y_test, y_pred)))

    import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_vs_numerical_features(df, numerical_features):
    """
    Plot box plots of numerical features separated by the 'churn' variable
    after removing outliers.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        numerical_features (list): A list of numerical feature names.
    """
    def remove_outliers(df, labels):
        """
        Remove outliers from the DataFrame using the Interquartile Range (IQR) method.
        
        Args:
            df (pandas.DataFrame): The input DataFrame.
            labels (list): A list of column names to remove outliers from.
            
        Returns:
            pandas.DataFrame: The DataFrame with outliers removed.
        """
        for label in labels:
            q1 = df[label].quantile(0.25)
            q3 = df[label].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr
            df[label] = df[label].mask(df[label] < lower_bound, df[label].median(), axis=0)
            df[label] = df[label].mask(df[label] > upper_bound, df[label].median(), axis=0)
        return df

    df = remove_outliers(df, numerical_features)

    for feature in numerical_features:
        if feature != 'churn':
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='churn', y=feature, data=df)
            plt.xlabel('Churn')
            plt.ylabel(feature)
            plt.title(f'Churn VS {feature}')
            plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_vs_numerical_features(df, numerical_features):
    """
    Plot box plots of numerical features separated by the 'churn' variable
    after removing outliers.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        numerical_features (list): A list of numerical feature names.
    """
    def remove_outliers(df, labels):
        """
        Remove outliers from the DataFrame using the Interquartile Range (IQR) method.
        
        Args:
            df (pandas.DataFrame): The input DataFrame.
            labels (list): A list of column names to remove outliers from.
            
        Returns:
            pandas.DataFrame: The DataFrame with outliers removed.
        """
        for label in labels:
            q1 = df[label].quantile(0.25)
            q3 = df[label].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr
            df[label] = df[label].mask(df[label] < lower_bound, df[label].median(), axis=0)
            df[label] = df[label].mask(df[label] > upper_bound, df[label].median(), axis=0)
        return df

    df = remove_outliers(df, numerical_features)

    for feature in numerical_features:
        if feature != 'churn':
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='churn', y=feature, data=df)
            plt.xlabel('Churn')
            plt.ylabel(feature)
            plt.title(f'Churn VS {feature}')
            plt.show()

#functions for removing outliers
def remove_outliers(df,labels):
    for label in labels:
        q1 = df[label].quantile(0.25)
        q3 = df[label].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        df[label] = df[label].mask(df[label]< lower_bound, df[label].median(),axis=0)
        df[label] = df[label].mask(df[label]> upper_bound, df[label].median(),axis=0)

    return df

import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_vs_numerical_features(df, numerical_features):
    """
    Plot box plots of numerical features separated by the 'churn' variable.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        numerical_features (list): A list of numerical feature names.
    """
    for feature in numerical_features:
        if feature != 'churn':
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='churn', y=feature, data=df)
            plt.xlabel('Churn')
            plt.ylabel(feature)
            plt.title(f'Churn VS {feature}')
            plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_numerical_feature_distributions(df, numerical_features):
    """
    Plot the density distribution of numerical features in the DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        numerical_features (list): A list of numerical feature names.
    """
    for feature in numerical_features:
        sns.distplot(df[feature])
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.title(f'Distribution of {feature}')
        plt.show()

def get_numerical_features(df):
    """
    Extract the numerical features from a DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        
    Returns:
        list: A list of numerical feature names.
    """
    numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
    return numerical_features

