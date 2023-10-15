# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

# %%
train_dataset = pd.read_csv('train.csv')
valid_dataset = pd.read_csv('valid.csv')
test_dataset = pd.read_csv('test.csv')

# %% [markdown]
# # Label 01

# %%
label_1_train_ori = train_dataset.drop(columns=['label_2','label_3','label_4'])

# %%
label_1_valid_ori = valid_dataset.drop(columns=['label_2','label_3','label_4'])

# %% [markdown]
# ## Handle Missing values

# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(label_1_train_ori.isnull().sum())

# %% [markdown]
# There's no missing values in label_1

# %% [markdown]
# ## Before Feature Engineering or Hyperparameter Tuning

# %%
X_train_ini = label_1_train_ori.drop(columns=['label_1'])
y_train_ini = label_1_train_ori['label_1']
X_valid_ori = label_1_valid_ori.drop(columns=['label_1'])
y_valid_ori = label_1_valid_ori['label_1']

clf = SVC(kernel='linear')
clf.fit(X_train_ini,y_train_ini)
predictions = clf.predict(X_valid_ori)

# %%
accuracy = accuracy_score(y_valid_ori, predictions)
print(f"Accuracy: {accuracy*100:.2f}%")

# %% [markdown]
# ## PCA Analysis

# %% [markdown]
# 

# %%
PCA_analysis_df = label_1_train_ori.drop(columns=['label_1'])

# %%
PCA_analysis_df_valid = label_1_valid_ori.drop(columns=['label_1'])

# %%
scaler = StandardScaler()
scaled_data = scaler.fit_transform(PCA_analysis_df)

# %%
scaled_valid_data = scaler.transform(PCA_analysis_df_valid)

# %%
pca = PCA(n_components=0.97,svd_solver='full')
principal_components = pca.fit_transform(scaled_data)

# %%
valid_principal_components = pca.transform(scaled_valid_data)

# %%
principal_df_label_1 = pd.DataFrame(data=principal_components, columns=[f'new_feature_{i}' for i in range(principal_components.shape[1])])
valid_principal_df_label_1 = pd.DataFrame(data = valid_principal_components,columns=[f'new_feature_{i}' for i in range(valid_principal_components.shape[1])])

# %% [markdown]
# ## Hyperparameter Tuning

# %%
param_dist = {
    'C': [0.1,1],           # Uniform distribution between 0 and 10
    'gamma': [0.001, 1],           # Log-uniform distribution between 0.001 and 1
    'kernel': ['rbf'],
    'degree': [1,2]
}

# %%
random_search = RandomizedSearchCV(
    SVC(),
    param_distributions=param_dist,
    n_iter=2,       # Number of parameter settings sampled
    cv=5,             # Number of cross-validation folds
    verbose=1,
    n_jobs=-1         # Use all CPU cores
)

# %%
X = principal_df_label_1
y = label_1_train_ori['label_1']

# %%
random_search.fit(X, y)

# %%
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# %% [markdown]
# ## Predicting using a SVC classifier

# %%
y_pred_label_1 = random_search.best_estimator_.predict(valid_principal_df_label_1)
accuracy = accuracy_score(y_pred_label_1, label_1_valid_ori['label_1'])
print(f"Accuracy on validation data: {accuracy*100:.2f}%")

# %% [markdown]
# ## Make predictions & Test dataset

# %%
data_to_predict = test_dataset.drop(columns=['ID'])
PCA_analysis_df_test_scaled = scaler.transform(data_to_predict)
test_principal_components = pca.transform(PCA_analysis_df_test_scaled)
test_principal_df = pd.DataFrame(data = test_principal_components,columns=[f'new_feature_{i}' for i in range(test_principal_components.shape[1])])

# %%
final_label_1_predict = random_search.best_estimator_.predict(test_principal_df)

# %%
final_submission = "190601D_submission1.csv"
data = {
 'ID' : test_dataset['ID'],
 'label_1' : final_label_1_predict
}
df = pd.DataFrame(data)
df.to_csv(final_submission,index=False)

# %% [markdown]
# ## Explainability of Results

# %%
def print_top_weight_features(svm_model, n_classes):
  coefficients = svm_model.coef_
  absolute_coefficients = np.abs(coefficients)
  top_weights = []
  n_classes = len(y.unique())
  for class_A in range(n_classes):
    for class_B in range(class_A + 1, n_classes):
        index = int(class_A * (2 * n_classes - class_A - 1) / 2 + class_B - class_A - 1)
        for feature_index, weight in enumerate(coefficients[index]):
            absolute_weight = np.abs(weight)
            if len(top_weights) < 20:
                heapq.heappush(top_weights, (absolute_weight, feature_index, class_A, class_B, weight))
            else:
                if absolute_weight > top_weights[0][0]:
                    heapq.heappop(top_weights)
                    heapq.heappush(top_weights, (absolute_weight, feature_index, class_A, class_B, weight))
  for i, (absolute_weight, feature_index, class_A, class_B, weight) in enumerate(reversed(top_weights)):
    print(f"Top {i + 1}: The weight assigned to feature {feature_index} between class {class_A} and class {class_B} is {weight}")


