# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# %%
train_dataset = pd.read_csv('train.csv')
valid_dataset = pd.read_csv('valid.csv')
test_dataset = pd.read_csv('test.csv')

# %% [markdown]
# # Label 01

# %%
label_1_train_ori = train_dataset.drop(columns=['label_2','label_3','label_4'])
label_1_valid_ori = valid_dataset.drop(columns=['label_2','label_3','label_4'])

# %% [markdown]
# ## Handle Missing Values

# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(label_1_train_ori.isnull().sum())

# %% [markdown]
# No missing values in label_1

# %% [markdown]
# ## Check for category distribution

# %%
value_counts = label_1_train_ori['label_1'].value_counts()
plt.figure(figsize=(10, 6))
value_counts.plot(kind='bar')
plt.title('Value Counts of Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# No need to handle class imbalances

# %% [markdown]
# ## Without any tuning

# %%
X_train_ini_label_1 = label_1_train_ori.drop(columns=['label_1'])
y_train_ini_label_1 = label_1_train_ori['label_1']
X_valid_ori_label_1 = label_1_valid_ori.drop(columns=['label_1'])
y_valid_ori_label_1 = label_1_valid_ori['label_1']

scaler = StandardScaler()
scaled_data_label_1 = scaler.fit_transform(X_train_ini_label_1)
scaled_valid_data_label_1 = scaler.transform(X_valid_ori_label_1)

# %%
clf = SVC(kernel='linear')
clf.fit(X_train_ini_label_1,y_train_ini_label_1)
predictions_label_1 = clf.predict(X_valid_ori_label_1)
initial_accuracy = accuracy_score(y_valid_ori_label_1, predictions_label_1)
print(f"Accuracy: {initial_accuracy*100:.2f}%")

# %%
label_1_data_to_predict = test_dataset.drop(columns=['ID'])
test_scaled_label_1 = scaler.transform(label_1_data_to_predict)
test_principal_df = pd.DataFrame(data = test_scaled_label_1,columns=[f'new_feature_{i}' for i in range(test_scaled_label_1.shape[1])])

final_label_1_predict = clf.predict(test_principal_df)

# %%
final_submission = "190601D_submission1.csv"
data = {
 'ID' : test_dataset['ID'],
 'label_1' : final_label_1_predict
}
df = pd.DataFrame(data)
df.to_csv(final_submission,index=False)

# %% [markdown]
# ## PCA Analysis

# %%
pca = PCA(n_components=0.97,svd_solver='full')
principal_components_label_1 = pca.fit_transform(scaled_data_label_1)
valid_principal_components_label_1 = pca.transform(scaled_valid_data_label_1)

principal_df_label_1 = pd.DataFrame(data=principal_components_label_1, columns=[f'new_feature_{i}' for i in range(principal_components_label_1.shape[1])])
valid_principal_df_label_1 = pd.DataFrame(data = valid_principal_components_label_1,columns=[f'new_feature_{i}' for i in range(valid_principal_components_label_1.shape[1])])

# %%
clf = SVC(kernel='linear')
clf.fit(principal_df_label_1,y_train_ini_label_1)
predictions_label_1 = clf.predict(valid_principal_df_label_1)
_accuracy = accuracy_score(y_valid_ori_label_1, predictions_label_1)
print(f"Accuracy: {_accuracy*100:.2f}%")

# %% [markdown]
# ## Hyperparameter Tuning

# %%
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

param_grid = {
                'C': [1, 10, 20, 30, 40, 50, 100],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
                }

base_estimator = SVC(gamma='scale', kernel='rbf', random_state=42)
search = HalvingGridSearchCV(base_estimator, param_grid, cv=5, verbose=1, n_jobs=7)
search.fit(principal_df_label_1, y_train_ini_label_1)

print(search.best_params_)
print(search.best_score_)

# %%
y_pred_label_1 = search.best_estimator_.predict(valid_principal_df_label_1)
accuracy_tuned = accuracy_score(y_pred_label_1, y_valid_ori_label_1)
print(f"Accuracy on validation data: {accuracy_tuned * 100:.2f}%")


