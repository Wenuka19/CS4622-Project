# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

# %%
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

# %%
train_dataset = pd.read_csv('train.csv')
valid_dataset = pd.read_csv('valid.csv')
test_dataset = pd.read_csv('test.csv')

# %% [markdown]
# # Label 03

# %%
label_3_train_ori = train_dataset.drop(columns=['label_1','label_2','label_4'])
label_3_valid_ori = valid_dataset.drop(columns=['label_1','label_2','label_4'])

# %% [markdown]
# ## Without Feature Engineering & Hyperparameter Tuning

# %%
X_train_ini_label_3 = label_3_train_ori.drop(columns=['label_3'])
y_train_ini_label_3 = label_3_train_ori['label_3']
X_valid_ori_label_3 = label_3_valid_ori.drop(columns=['label_3'])
y_valid_ori_label_3 = label_3_valid_ori['label_3']

clf = SVC(kernel='linear')
clf.fit(X_train_ini_label_3,y_train_ini_label_3)
predictions = clf.predict(X_valid_ori_label_3)

# %%
accuracy = accuracy_score(y_valid_ori_label_3, predictions)
print(f"Accuracy: {accuracy*100:.2f}%")

# %% [markdown]
# ## Handle class imbalances

# %%
value_counts = label_3_train_ori['label_3'].value_counts()
plt.figure(figsize=(10, 6))
value_counts.plot(kind='bar')
plt.title('Value Counts of Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# There's a considerable difference between values between two categories

# %%
label_3_train_ori_X= label_3_train_ori.drop(columns=['label_3'])
label_3_train_ori_y = label_3_train_ori['label_3']
ros = RandomOverSampler(random_state=42)
X_resampled_label_3, y_resampled_label_3 = ros.fit_resample(label_3_train_ori_X, label_3_train_ori_y)

# %%
value_counts = y_resampled_label_3.value_counts()
plt.figure(figsize=(10, 6))
value_counts.plot(kind='bar')
plt.title('Value Counts of Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# ## PCA Analysis

# %%
scaler = StandardScaler()
scaled_data_label_3 = scaler.fit_transform(X_resampled_label_3)
scaled_valid_data_label_3 = scaler.transform(X_valid_ori_label_3)

pca = PCA(n_components=0.96,svd_solver='full')
principal_components_label_3 = pca.fit_transform(scaled_data_label_3)
valid_principal_components_label_3 = pca.transform(scaled_valid_data_label_3)

principal_df_label_3 = pd.DataFrame(data=principal_components_label_3, columns=[f'new_feature_{i}' for i in range(principal_components_label_3.shape[1])])
valid_principal_df_label_3 = pd.DataFrame(data = valid_principal_components_label_3,columns=[f'new_feature_{i}' for i in range(valid_principal_components_label_3.shape[1])])


# %%
clf_1 = SVC()
clf_1.fit(principal_df_label_3,y_resampled_label_3)
predictions_label_3 = clf_1.predict(valid_principal_df_label_3)

accuracy = accuracy_score(y_valid_ori_label_3, predictions_label_3)
print(f"Accuracy: {accuracy*100:.2f}%")

# %%
label__data_to_predict = test_dataset.drop(columns=['ID'])
PCA_analysis_df_test_scaled_label_3 = scaler.transform(label__data_to_predict)
test_principal_components_label_3 = pca.transform(PCA_analysis_df_test_scaled_label_3)
test_principal_df = pd.DataFrame(data = test_principal_components_label_3,columns=[f'new_feature_{i}' for i in range(test_principal_components_label_3.shape[1])])

final_label_3_predict = clf_1.predict(test_principal_df)

# %%
final_submission = "190601D_submission1.csv"
dataframe = pd.read_csv(final_submission)
dataframe['label_3'] = final_label_3_predict
dataframe.to_csv(final_submission,index=False)

# %% [markdown]
# ## Hyperparameter Tuning

# %%
param_dist_label_3 = {
    'C': uniform(loc=0, scale=10),           
    'gamma': reciprocal(0.001, 1),           
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4, 5, 6]               
}

random_search = RandomizedSearchCV(
    SVC(),
    param_distributions=param_dist_label_3,
    n_iter=10,       # Number of parameter settings sampled
    cv=5,             # Number of cross-validation folds
    verbose=1,
    n_jobs=-1         # Use all CPU cores
)

random_search.fit(principal_df_label_3, y_resampled_label_3)



