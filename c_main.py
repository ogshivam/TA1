from c_data_preprocessing import load_data, preprocess_data
#from c_all_function import get_numerical_features , plot_numerical_feature_distributions , plot_churn_vs_numerical_features , remove_outliers , plot_churn_vs_numerical_features , plot_churn_vs_numerical_features , train_and_evaluate_svc , train_and_evaluate_random_forest , logistic_regression_with_roc_auc , plot_roc_curve , plot_roc_curve , plot_roc_curves , plot_roc_curves , plot_confusion_matrix , calculate_prediction_percentages , 
from c_all_function import *

df = load_data("Customer_Churn_Prediction.csv")
df = preprocess_data(df)

# Get the list of categorical features
cat_features = get_categorical_features(df)
print('List of categorical variables:', cat_features)

# Get the list of categorical features
categorical_features = get_categorical_features(df)

# Display count plots for each categorical variable
display_categorical_variables(df, categorical_features)

# Get the list of numerical features
num_features = get_numerical_features(df)
print('List of Numerical features:', num_features)

# Get the list of numerical features
numerical_features = get_numerical_features(df)

# Plot the density distribution of numerical features
plot_numerical_feature_distributions(df, numerical_features)

# Get the list of numerical features
numerical_features = get_numerical_features(df)

# Plot the distribution of numerical features separated by the 'churn' variable
plot_churn_rate_vs_numerical_features(df, numerical_features)

# Get the list of numerical features
numerical_features = get_numerical_features(df)

# Plot box plots of numerical features separated by the 'churn' variable
plot_churn_vs_numerical_features(df, numerical_features)

df = remove_outliers(df, numerical_features)

# Get the list of numerical features
numerical_features = get_numerical_features(df)

# Plot box plots of numerical features separated by the 'churn' variable
plot_churn_vs_numerical_features(df, numerical_features)

hash_state = ce.HashingEncoder(cols = 'state')
train = hash_state.fit_transform(df)
test = hash_state.transform(df)
train.head()

# replace no to 0 and yes to 1
train.international_plan.replace(['no','yes'],[0,1],inplace = True)
train.voice_mail_plan.replace(['no','yes'],[0,1],inplace=True)
train.churn.replace(['no','yes'],[0,1],inplace = True)
test.international_plan.replace(['no','yes'],[0,1],inplace = True)
test.voice_mail_plan.replace(['no','yes'],[0,1],inplace = True)
train.head()

# converting the area_code to numerical variable using one-hot encoder
onehot_area = OneHotEncoder()
onehot_area.fit(train[['area_code']])

# Train
encoded_values = onehot_area.transform(train[['area_code']])
train[onehot_area.categories_[0]] = encoded_values.toarray()
train = train.drop('area_code', axis=1)

# Test
encoded_values = onehot_area.transform(test[['area_code']])
test[onehot_area.categories_[0]] = encoded_values.toarray()
test = test.drop('area_code', axis=1)

# showing the imbalanced class
sns.countplot(x = 'churn', data = train)
plt.show()

from sklearn.model_selection import train_test_split
x = train.drop('churn',axis=1).values
y = train.churn.values
id_submission = test.churn
test = test.drop('churn', axis=1)
# spliting the data into test and train
x_train, x_test , y_train, y_test = train_test_split(x, y , test_size=0.3, random_state=0)

id_submission = df['churn']

print('Before upsampling count of label 0 {}'.format(sum(y_train==0)))
print('Before upsampling count of label 1 {}'.format(sum(y_train==1)))
# Minority Over Sampling Technique
sm = SMOTE(sampling_strategy = 1, random_state=1)
x_train_s, y_train_s = sm.fit_resample(x_train, y_train.ravel())

print('After upsampling count of label 0 {}'.format(sum(y_train_s==0)))
print('After upsampling count of label 1 {}'.format(sum(y_train_s==1)))

# creating the object of minmax scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)

# Train and evaluate the SVC model
train_and_evaluate_svc(X_train, X_test, y_train, y_test)

# Train and evaluate the Random Forest model
train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)

# Train and evaluate the XGBoost model
train_and_evaluate_xgboost(X_train, X_test, y_train, y_test)

y_pred_sub = clf.predict(test)

submit = pd.DataFrame({'id':id_submission, 'churn1':y_pred_sub})
submit.head()

# replace 0 to no and 1 to yes
submit.churn1.replace([0,1],['no','yes'], inplace=True)

submit.to_csv('churn_submit.csv',index=False)

from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for the test set
y_pred_prob = clf.predict_proba(x_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Print AUC
print("Area under the ROC curve:", roc_auc)

# Usage example:
plot_roc_curve(rfc, x_test, y_test, "Random Forest")

# Usage example:
svc = SVC(probability=True)
svc.fit(x_train, y_train)
plot_roc_curve(svc, x_test, y_test, "SVM")

# Usage example:
logistic_regression_with_roc_auc(x_train, y_train, x_test, y_test)

#Usage example:
models = [svc, rfc, clf]
model_names = ["SVM", "Random Forest", "XGBoost"]
plot_roc_curves(models, model_names, x_test, y_test)

# Usage example:
models = [svc, rfc, clf]
model_names = ["SVM", "Random Forest", "XGBoost"]
plot_roc_curves(models, model_names, x_test, y_test)

# Print the accuracy for each model
print("SVM accuracy:", accuracy_score(y_test, svc.predict(x_test)))
print("Random Forest accuracy:", accuracy_score(y_test, rfc.predict(x_test)))
print("XGBoost accuracy:", accuracy_score(y_test, clf.predict(x_test)))

# Usage example:
generate_and_plot_confusion_matrix(svc, x_test, y_test, 'SVM')
generate_and_plot_confusion_matrix(rfc, x_test, y_test, 'Random Forest')
generate_and_plot_confusion_matrix(clf, x_test, y_test, 'XGBoost')
generate_and_plot_confusion_matrix(logreg, x_test, y_test, 'Logistic Regression')

calculate_prediction_percentages(confusion_matrices)






