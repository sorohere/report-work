# This cell implements a complete predictive modeling pipeline for telecom customer churn
# using the processed data from the previous cleaning steps.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Define feature columns and target
# We'll exclude accountid as identifier and use 'churn' as the target.

# We assume that churn is a binary column already present
# Identify numerical and categorical columns based on dataset columns

# Our current cleaned data 'data' is already in memory
# Let's select features; assume these columns (this can be extended as needed):
features = ['tenure', 'city_tier', 'cc_contacted_ly', 'service_score', 'account_user_count', 'cc_agent_score', 'complain_ly', 'rev_per_month', 'rev_growth_yoy']
# Additional categorical features: 'payment', 'gender', 'account_segment', 'marital_status', 'login_device'
cat_features = ['payment', 'gender', 'account_segment', 'marital_status', 'login_device']

X = data[features + cat_features]
y = data['churn']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Preprocessing pipeline

# Numeric pipeline: imputation (if any missing) and scaling
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: imputation and one-hot encoding
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, features),
    ('cat', categorical_pipeline, cat_features)
])

# Define a function to perform modeling and evaluation

def evaluate_model(model, model_name):
    # Create pipeline with the preprocessor and model
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])
    
    # Use GridSearchCV for hyperparameter tuning (if applicable)
    # Define parameter grids for each type of model
    if model_name == 'LogisticRegression':
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs']
        }
    elif model_name == 'RandomForest':
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5]
        }
    elif model_name == 'XGBoost':
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
    else:
        param_grid = {}
    
    # Set up cross-validation:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(model_name + ' best parameters:', grid.best_params_)
    
    # Predict on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(model_name + ' Metrics:')
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    print('ROC-AUC:', roc_auc)
    print('\
Classification Report:\
', classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9,6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=.5, linecolor='#333333')
    plt.title(model_name + ' Confusion Matrix', fontsize=20, pad=15, color='#222222')
    plt.xlabel('Predicted Label', fontsize=16, labelpad=10, color='#333333')
    plt.ylabel('True Label', fontsize=16, labelpad=10, color='#333333')
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(9,6))
    plt.plot(fpr, tpr, color='#766CDB', label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='#52515E', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=16, labelpad=10, color='#333333')
    plt.ylabel('True Positive Rate', fontsize=16, labelpad=10, color='#333333')
    plt.title(model_name + ' ROC Curve', fontsize=20, pad=15, color='#222222')
    plt.legend(loc='lower right', fontsize=12, frameon=False, labelcolor='#333333')
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    plt.show()
    
    # Return best model pipeline and predictions for further analysis
    return best_model, y_pred, y_proba

# Evaluate

# Install XGBoost
%pip install xgboost
print("XGBoost installed successfully.")

# Load the data
import pandas as pd
import numpy as np

# Load the telecom customer dataset
data = pd.read_csv('data.csv')

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\
First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\
Missing values per column:")
print(data.isnull().sum())

# Data Cleaning and Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set the style for plots
plt.style.use('default')

# Load the data
data = pd.read_csv('data.csv')

# Check data types
print("Data types:")
print(data.dtypes)

# Basic statistics
print("\
Basic statistics:")
print(data.describe())


# Data Cleaning and Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Set the plotting style
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.titlepad'] = 15
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['axes.axisbelow'] = True

# Load the data
data = pd.read_csv('data.csv')

# Convert tenure to numeric (handling errors)
data['tenure'] = pd.to_numeric(data['tenure'], errors='coerce')

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values after converting tenure to numeric:")
print(missing_values)

# Fill missing values
# For numerical columns, use median
numerical_cols = ['tenure', 'city_tier', 'cc_contacted_ly', 'service_score', 
                 'account_user_count', 'cc_agent_score', 'rev_per_month', 
                 'complain_ly', 'day_since_cc_connect', 'cashback']

for col in numerical_cols:
    data[col] = data[col].fillna(data[col].median())

# For categorical columns, use mode
categorical_cols = ['payment', 'gender', 'account_segment', 'marital_status', 'login_device']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Check if missing values are handled
print("\
Missing values after imputation:")
print(data.isnull().sum())

# Display the cleaned data
print("\
Cleaned data sample:")
print(data.head())

# Basic statistics of cleaned data
print("\
Basic statistics after cleaning:")
print(data.describe())


# Let's first check the data types and values in problematic columns
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Check the unique values in service_score to understand the issue
print("Service score unique values:")
print(data['service_score'].unique())

# Check account_user_count values
print("\
Account user count unique values:")
print(data['account_user_count'].unique())

# Display the shape of the dataset
print("\
Dataset shape:", data.shape)

# Display the first few rows
print("\
First few rows:")
print(data.head())


# Fix account_user_count: Remove non-digit values and convert
import re

def clean_account_user_count(x):
    try:
        # Remove non-digit characters
        cleaned = re.sub('[^0-9]', '', str(x))
        if cleaned == '':
            return np.nan
        else:
            return float(cleaned)
    except Exception as e:
        return np.nan

# Apply cleaning to account_user_count column

data['account_user_count'] = data['account_user_count'].apply(clean_account_user_count)

# Convert other columns to numeric where applicable
numeric_columns = ['tenure', 'city_tier', 'cc_contacted_ly', 'service_score', 'account_user_count', 
                   'cc_agent_score', 'rev_per_month', 'complain_ly', 'day_since_cc_connect', 'cashback']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Impute missing values for numeric columns with median
for col in numeric_columns:
    data[col].fillna(data[col].median(), inplace=True)

# Impute categorical missing values with mode for specified columns
categorical_cols = ['payment', 'gender', 'account_segment', 'marital_status', 'login_device']
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Our target variable is churn
# Define feature columns (exclude accountid and churn)
feature_cols = [col for col in data.columns if col not in ['accountid','churn']]

# Separate features and target
X = data[feature_cols]
y = data['churn']

# We'll need to encode categorical columns and scale numeric columns
numeric_features = numeric_columns  
# consider payment, gender, account_segment, marital_status, login_device as categorical
categorical_features = categorical_cols

# Build column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Build pipeline with logistic regression as a baseline model
from sklearn.linear_model import LogisticRegression
pipe_lr = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(max_iter=1000))])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Train-test split done.')

# Fit logistic regression model
pipe_lr.fit(X_train, y_train)

# Predict on test data
y_pred_lr = pipe_lr.predict(X_test)

# Evaluate logistic regression model
acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_lr = roc_auc_score(y_test, pipe_lr.predict_proba(X_test)[:,1])

print('Logistic Regression Performance on Test Set:')
print('Accuracy:', acc_lr)
print('Precision:', prec_lr)
print('Recall:', recall_lr)
print('F1 Score:', f1_lr)
print('ROC AUC:', roc_lr)

# Do cross-validation for logistic regression
cv_scores_lr = cross_val_score(pipe_lr, X_train, y_train, cv=5, scoring='accuracy')
print('\
Logistic Regression Cross-Validation Accuracy Scores:', cv_scores_lr)

# Also, build a RandomForestClassifier and XGBoost model as alternatives

# Random Forest Pipeline
from sklearn.ensemble import RandomForestClassifier
pipe_rf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])
pipe_rf.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = pipe_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_rf = roc_auc_score(y_test, pipe_rf.predict_proba(X_test)[:,1])

print('\
Random Forest Performance on Test Set:')
print('Accuracy:', acc_rf)
print('Precision:', prec_rf)
print('Recall:', recall_rf)
print('F1 Score:', f1_rf)
print('ROC AUC:', roc_rf)

cv_scores_rf = cross_val_score(pipe_rf, X_train, y_train, cv=5, scoring='accuracy')
print('\
Random Forest Cross-Validation Accuracy Scores:', cv_scores_rf)

# XGBoost Pipeline
pipe_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
pipe_xgb.fit(X_train, y_train)

# Evaluate XGBoost
y_pred_xgb = pipe_xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_xgb = roc_auc_score(y_test, pipe_xgb.predict_proba(X_test)[:,1])

print('\
XGBoost Performance on Test Set:')
print('Accuracy:', acc_xgb)
print('Precision:', prec_xgb)
print('Recall:', recall_xgb)
print('F1 Score:', f1_xgb)
print('ROC AUC:', roc_xgb)

cv_scores_xgb = cross_val_score(pipe_xgb, X_train, y_train, cv=5, scoring='accuracy')
print('\
XGBoost Cross-Validation Accuracy Scores:', cv_scores_xgb)

print('\
Model building and evaluation complete.')

print('done')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import matplotlib.font_manager as fm

# Set the style for plots
plt.style.use('default')

# Load the data
data = pd.read_csv('data.csv')

# Fix account_user_count: Remove non-digit values and convert
import re

def clean_account_user_count(x):
    try:
        # Remove non-digit characters
        cleaned = re.sub('[^0-9]', '', str(x))
        if cleaned == '':
            return np.nan
        else:
            return float(cleaned)
    except Exception as e:
        return np.nan

# Apply cleaning to account_user_count column
data['account_user_count'] = data['account_user_count'].apply(clean_account_user_count)

# Convert other columns to numeric where applicable
numeric_columns = ['tenure', 'city_tier', 'cc_contacted_ly', 'service_score', 'account_user_count', 
                   'cc_agent_score', 'rev_per_month', 'complain_ly', 'day_since_cc_connect', 'cashback']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Impute missing values for numeric columns with median
for col in numeric_columns:
    data[col].fillna(data[col].median(), inplace=True)

# Impute categorical missing values with mode for specified columns
categorical_cols = ['payment', 'gender', 'account_segment', 'marital_status', 'login_device']
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Check churn distribution
churn_counts = data['churn'].value_counts()
print("Churn Distribution:")
print(churn_counts)
print(f"Churn Rate: {churn_counts[1] / len(data) * 100:.2f}%")

# Create a figure for churn distribution
fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
ax.set_axisbelow(True)

# Plot churn distribution
sns.countplot(x='churn', data=data, palette=['#766CDB', '#DA847C'], ax=ax)
ax.set_title('Churn Distribution', fontsize=20, fontweight='semibold', color='#222222', pad=15)
ax.set_xlabel('Churn (0=No, 1=Yes)', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax.set_ylabel('Count', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax.tick_params(axis='both', labelsize=14, colors='#555555')
ax.spines['top'].set_color('#333333')
ax.spines['right'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.spines['left'].set_color('#333333')
ax.grid(axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')

# Add percentage labels
for i, count in enumerate(churn_counts):
    percentage = count / len(data) * 100
    ax.text(i, count + 100, f'{percentage:.1f}%', ha='center', fontsize=14, color='#333333')

plt.tight_layout()
plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a figure for numeric features correlation with churn
plt.figure(figsize=(12, 10))
corr = data[numeric_columns + ['churn']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
ax.set_axisbelow(True)

# Plot correlation heatmap
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            linewidths=0.5, ax=ax, annot_kws={"size": 12, "color": "#333333"})
ax.set_title('Correlation of Numeric Features with Churn', fontsize=20, fontweight='semibold', color='#222222', pad=15)
ax.tick_params(axis='both', labelsize=14, colors='#555555')

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Prepare data for model evaluation
# Define feature columns (exclude accountid and churn)
feature_cols = [col for col in data.columns if col not in ['accountid','churn']]

# Separate features and target
X = data[feature_cols]
y = data['churn']

# We'll need to encode categorical columns and scale numeric columns
numeric_features = numeric_columns  
categorical_features = categorical_cols

# Build column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build and fit models
# Logistic Regression
pipe_lr = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(max_iter=1000))])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
y_prob_lr = pipe_lr.predict_proba(X_test)[:,1]

# Random Forest
pipe_rf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=42))])
pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)
y_prob_rf = pipe_rf.predict_proba(X_test)[:,1]

# XGBoost
pipe_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
pipe_xgb.fit(X_train, y_train)
y_pred_xgb = pipe_xgb.predict(X_test)
y_prob_xgb = pipe_xgb.predict_proba(X_test)[:,1]

# Create a figure for model comparison
fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
ax.set_axisbelow(True)

# Metrics for comparison
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracy = [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_xgb)]
precision = [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_rf), precision_score(y_test, y_pred_xgb)]
recall = [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_rf), recall_score(y_test, y_pred_xgb)]
f1 = [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_xgb)]
roc_auc = [roc_auc_score(y_test, y_prob_lr), roc_auc_score(y_test, y_prob_rf), roc_auc_score(y_test, y_prob_xgb)]

# Create a DataFrame for plotting
metrics_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC AUC': roc_auc
})

# Melt the DataFrame for easier plotting
metrics_melted = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Score')

# Plot model comparison
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_melted, palette=['#766CDB', '#DA847C', '#D9CC8B', '#7CD9A5', '#877877'], ax=ax)
ax.set_title('Model Performance Comparison', fontsize=20, fontweight='semibold', color='#222222', pad=15)
ax.set_xlabel('Model', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax.set_ylabel('Score', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax.tick_params(axis='both', labelsize=14, colors='#555555')
ax.spines['top'].set_color('#333333')
ax.spines['right'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.spines['left'].set_color('#333333')
ax.grid(axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot ROC curves
fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
ax.set_axisbelow(True)

# Calculate ROC curve for each model
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

# Plot ROC curves
plt.plot(fpr_lr, tpr_lr, color='#766CDB', lw=2, label=f'Logistic Regression (AUC = {roc_auc[0]:.3f})')
plt.plot(fpr_rf, tpr_rf, color='#DA847C', lw=2, label=f'Random Forest (AUC = {roc_auc[1]:.3f})')
plt.plot(fpr_xgb, tpr_xgb, color='#D9CC8B', lw=2, label=f'XGBoost (AUC = {roc_auc[2]:.3f})')
plt.plot([0, 1], [0, 1], color='#877877', lw=2, linestyle='--', label='Random Guess')

ax.set_title('ROC Curves for Different Models', fontsize=20, fontweight='semibold', color='#222222', pad=15)
ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax.tick_params(axis='both', labelsize=14, colors='#555555')
ax.spines['top'].set_color('#333333')
ax.spines['right'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.spines['left'].set_color('#333333')
ax.grid(linestyle='--', alpha=0.7, color='#E0E0E0')
ax.legend(fontsize=12, loc='lower right')

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance for Random Forest (best performing model)
# Get feature names after preprocessing
cat_features = pipe_rf.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, cat_features])

# Get feature importances
importances = pipe_rf.named_steps['classifier'].feature_importances_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(15)

# Plot feature importances
fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
ax.set_axisbelow(True)

sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='Blues_r', ax=ax)
ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=20, fontweight='semibold', color='#222222', pad=15)
ax.set_xlabel('Importance', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax.set_ylabel('Feature', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax.tick_params(axis='both', labelsize=14, colors='#555555')
ax.spines['top'].set_color('#333333')
ax.spines['right'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.spines['left'].set_color('#333333')
ax.grid(axis='x', linestyle='--', alpha=0.7, color='#E0E0E0')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion matrices for all models
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)

# Confusion matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], annot_kws={"size": 14, "color": "#333333"})
axes[0].set_title('Logistic Regression\
Confusion Matrix', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[0].set_xlabel('Predicted', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[0].set_ylabel('Actual', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[0].tick_params(axis='both', labelsize=14, colors='#555555')
axes[0].set_xticklabels(['No Churn', 'Churn'])
axes[0].set_yticklabels(['No Churn', 'Churn'])

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1], annot_kws={"size": 14, "color": "#333333"})
axes[1].set_title('Random Forest\
Confusion Matrix', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[1].set_xlabel('Predicted', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[1].set_ylabel('Actual', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[1].tick_params(axis='both', labelsize=14, colors='#555555')
axes[1].set_xticklabels(['No Churn', 'Churn'])
axes[1].set_yticklabels(['No Churn', 'Churn'])

# Confusion matrix for XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[2], annot_kws={"size": 14, "color": "#333333"})
axes[2].set_title('XGBoost\
Confusion Matrix', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[2].set_xlabel('Predicted', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[2].set_ylabel('Actual', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[2].tick_params(axis='both', labelsize=14, colors='#555555')
axes[2].set_xticklabels(['No Churn', 'Churn'])
axes[2].set_yticklabels(['No Churn', 'Churn'])

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze churn by categorical features
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Churn by payment method
sns.countplot(x='payment', hue='churn', data=data, palette=['#766CDB', '#DA847C'], ax=axes[0, 0])
axes[0, 0].set_title('Churn by Payment Method', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[0, 0].set_xlabel('Payment Method', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[0, 0].set_ylabel('Count', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[0, 0].tick_params(axis='both', labelsize=14, colors='#555555')
axes[0, 0].legend(title='Churn', fontsize=12, title_fontsize=14)
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

# Churn by gender
sns.countplot(x='gender', hue='churn', data=data, palette=['#766CDB', '#DA847C'], ax=axes[0, 1])
axes[0, 1].set_title('Churn by Gender', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[0, 1].set_xlabel('Gender', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[0, 1].set_ylabel('Count', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[0, 1].tick_params(axis='both', labelsize=14, colors='#555555')
axes[0, 1].legend(title='Churn', fontsize=12, title_fontsize=14)

# Churn by account segment
sns.countplot(x='account_segment', hue='churn', data=data, palette=['#766CDB', '#DA847C'], ax=axes[1, 0])
axes[1, 0].set_title('Churn by Account Segment', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[1, 0].set_xlabel('Account Segment', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[1, 0].set_ylabel('Count', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[1, 0].tick_params(axis='both', labelsize=14, colors='#555555')
axes[1, 0].legend(title='Churn', fontsize=12, title_fontsize=14)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

# Churn by login device
sns.countplot(x='login_device', hue='churn', data=data, palette=['#766CDB', '#DA847C'], ax=axes[1, 1])
axes[1, 1].set_title('Churn by Login Device', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[1, 1].set_xlabel('Login Device', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[1, 1].set_ylabel('Count', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[1, 1].tick_params(axis='both', labelsize=14, colors='#555555')
axes[1, 1].legend(title='Churn', fontsize=12, title_fontsize=14)

plt.tight_layout()
plt.savefig('churn_by_categorical.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze churn by numeric features
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Churn by tenure
sns.boxplot(x='churn', y='tenure', data=data, palette=['#766CDB', '#DA847C'], ax=axes[0, 0])
axes[0, 0].set_title('Churn by Tenure', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[0, 0].set_xlabel('Churn', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[0, 0].set_ylabel('Tenure', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[0, 0].tick_params(axis='both', labelsize=14, colors='#555555')

# Churn by service score
sns.boxplot(x='churn', y='service_score', data=data, palette=['#766CDB', '#DA847C'], ax=axes[0, 1])
axes[0, 1].set_title('Churn by Service Score', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[0, 1].set_xlabel('Churn', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[0, 1].set_ylabel('Service Score', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[0, 1].tick_params(axis='both', labelsize=14, colors='#555555')

# Churn by revenue per month
sns.boxplot(x='churn', y='rev_per_month', data=data, palette=['#766CDB', '#DA847C'], ax=axes[1, 0])
axes[1, 0].set_title('Churn by Revenue per Month', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[1, 0].set_xlabel('Churn', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[1, 0].set_ylabel('Revenue per Month', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[1, 0].tick_params(axis='both', labelsize=14, colors='#555555')

# Churn by cashback
sns.boxplot(x='churn', y='cashback', data=data, palette=['#766CDB', '#DA847C'], ax=axes[1, 1])
axes[1, 1].set_title('Churn by Cashback', fontsize=20, fontweight='semibold', color='#222222', pad=15)
axes[1, 1].set_xlabel('Churn', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[1, 1].set_ylabel('Cashback', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
axes[1, 1].tick_params(axis='both', labelsize=14, colors='#555555')

plt.tight_layout()
plt.savefig('churn_by_numeric.png', dpi=300, bbox_inches='tight')
plt.show()

print("All visualizations have been created and saved.")
