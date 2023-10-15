# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import GridSearchCV

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

# %%
train_dataset = pd.read_csv('train.csv')
valid_dataset = pd.read_csv('valid.csv')
test_dataset = pd.read_csv('test.csv')

# %%
from catboost import CatBoostClassifier, Pool

# %% [markdown]
# # Label 02

# %%
label_2_train_ori = train_dataset.drop(columns=['label_1','label_3','label_4'])
label_2_valid_ori = valid_dataset.drop(columns=['label_1','label_3','label_4'])

# %% [markdown]
# ## Handle Missing values

# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(label_2_train_ori.isnull().sum())

# %% [markdown]
# There's 480 values missing in label_2. Since there's enough data to work with I'll drop the missing values

# %%
label_2_train_cleaned = label_2_train_ori.dropna()
label_2_valid_cleaned = label_2_valid_ori.dropna()

# %% [markdown]
# ## Without Feature Engineering or Hyperparameter tuning

# %%
X_train_label_2_ini = label_2_train_cleaned.drop(columns=['label_2'])
y_train_label_2_ini = label_2_train_cleaned['label_2']
X_valid_label_2_ini = label_2_valid_cleaned.drop(columns=['label_2'])
y_valid_label_2_ini = label_2_valid_cleaned['label_2']

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_label_2_ini)
X_valid_scaled = scaler.transform(X_valid_label_2_ini)

# %%
model = SVC()
model.fit(X_train_scaled,y_train_label_2_ini)
y_pred_label_2 = model.predict(X_valid_scaled)
accuracy = accuracy_score(y_valid_label_2_ini, y_pred_label_2)
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
label_2_data_to_predict = test_dataset.drop(columns=['ID'])
PCA_analysis_df_test_scaled_label_2 = scaler.transform(label_2_data_to_predict)

final_label_2_predict = model.predict(PCA_analysis_df_test_scaled_label_2)
final_submission = "190601D_submission1.csv"
dataframe = pd.read_csv(final_submission)
dataframe['label_2'] = final_label_2_predict
dataframe.to_csv(final_submission,index=False)

# %% [markdown]
# ## Feature selection using K-best

# %%
k_best = SelectKBest(score_func=f_classif, k=400)
X_train_selected = k_best.fit_transform(X_train_label_2_ini, y_train_label_2_ini)
X_valid_selected = k_best.transform(X_valid_scaled)

# %% [markdown]
# ## Feature selection using PCA

# %%
pca = PCA(n_components=0.97,svd_solver='full')
principal_components_label_2 = pca.fit_transform(X_train_scaled)
valid_principal_components_label_2 = pca.transform(X_valid_scaled)

# %%
principal_df_label_2 = pd.DataFrame(data=principal_components_label_2, columns=[f'new_feature_{i}' for i in range(principal_components_label_2.shape[1])])
valid_principal_df_label_2 = pd.DataFrame(data = valid_principal_components_label_2,columns=[f'new_feature_{i}' for i in range(valid_principal_components_label_2.shape[1])])

# %%
model_2 = SVC()
model_2.fit(principal_df_label_2,y_train_label_2_ini)
y_pred = model_2.predict(valid_principal_df_label_2)
accuracy = accuracy_score(y_valid_label_2_ini, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
label_2_data_to_predict = test_dataset.drop(columns=['ID'])
PCA_analysis_df_test_scaled_label_2 = scaler.transform(label_2_data_to_predict)
test_principal_components_label_2 = pca.transform(PCA_analysis_df_test_scaled_label_2)
test_principal_df = pd.DataFrame(data = test_principal_components_label_2,columns=[f'new_feature_{i}' for i in range(test_principal_components_label_2.shape[1])])

final_label_2_predict = model_2.predict(test_principal_df)

# %%
final_submission = "190601D_submission1.csv"
dataframe = pd.read_csv(final_submission)
dataframe['label_2'] = final_label_2_predict
dataframe.to_csv(final_submission,index=False)

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
search.fit(principal_df_label_2, y_train_label_2_ini)

print(search.best_params_)
print(search.best_score_)

# %%
valid_principal_df_label_2.shape

# %%
y_pred_label_2 = search.best_estimator_.predict(valid_principal_df_label_2)
accuracy_tuned = accuracy_score(y_pred_label_2, y_valid_label_2_ini)
print(f"Accuracy on validation data: {accuracy_tuned * 100:.2f}%")

# %% [markdown]
# ## Make predictions & Test dataset

# %%
label_2_data_to_predict = test_dataset.drop(columns=['ID'])
PCA_analysis_df_test_scaled_label_2 = scaler.transform(label_2_data_to_predict)
test_principal_components_label_2 = pca.transform(PCA_analysis_df_test_scaled_label_2)
test_principal_df = pd.DataFrame(data = test_principal_components_label_2,columns=[f'new_feature_{i}' for i in range(test_principal_components_label_2.shape[1])])

final_label_2_predict = search.best_estimator_.predict(test_principal_df)

# %%
final_submission = "190601D_submission1.csv"
dataframe = pd.read_csv(final_submission)
dataframe['label_2'] = final_label_2_predict
dataframe.to_csv(final_submission,index=False)


