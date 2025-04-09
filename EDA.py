# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plotting style
plt.style.use('seaborn-v0_8')

# Load data
# The file encoding is ascii as specified in metadata

data = pd.read_csv('data.csv', encoding='ascii')

# Clean column names (strip spaces if any)
data.columns = [col.strip() for col in data.columns]

# Print head of dataframe for confirmation
print(data.head())

# ------------------------------------
# 1. Boxplot: revenue distributions (rev_per_month) across customer segments (account_segment)

plt.figure(figsize=(9,6))
ax = sns.boxplot(x='account_segment', y='rev_per_month', data=data, palette=['#766CDB', '#DA847C', '#D9CC8B', '#7CD9A5', '#877877', '#52515E'])
ax.set_title('Revenue per Month by Account Segment', fontsize=20, pad=15, color='#222222')
ax.set_xlabel('Account Segment', fontsize=16, labelpad=10, color='#333333')
ax.set_ylabel('Revenue per Month', fontsize=16, labelpad=10, color='#333333')
ax.tick_params(axis='x', labelsize=14, colors='#555555')
ax.tick_params(axis='y', labelsize=14, colors='#555555')
ax.set_axisbelow(True)
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.show()

# ------------------------------------
# 2. Scatter Plot: tenure vs churn

plt.figure(figsize=(9,6))
ax = sns.scatterplot(x='tenure', y='churn', data=data, color='#766CDB', s=100)
ax.set_title('Scatter Plot of Tenure vs Churn', fontsize=20, pad=15, color='#222222')
ax.set_xlabel('Tenure (months)', fontsize=16, labelpad=10, color='#333333')
ax.set_ylabel('Churn', fontsize=16, labelpad=10, color='#333333')
ax.tick_params(axis='x', labelsize=14, colors='#555555')
ax.tick_params(axis='y', labelsize=14, colors='#555555')
ax.set_axisbelow(True)
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.show()

# ------------------------------------
# 3. Bar Graph: churn rates by city tier and payment methods
# Calculate churn rate per group
churn_by_tier_payment = data.groupby(['city_tier', 'payment'])['churn'].mean().reset_index()

plt.figure(figsize=(9,6))
ax = sns.barplot(x='city_tier', y='churn', hue='payment', data=churn_by_tier_payment, palette='muted')
ax.set_title('Average Churn Rate by City Tier and Payment Method', fontsize=20, pad=15, color='#222222')
ax.set_xlabel('City Tier', fontsize=16, labelpad=10, color='#333333')
ax.set_ylabel('Average Churn Rate', fontsize=16, labelpad=10, color='#333333')
ax.tick_params(axis='x', labelsize=14, colors='#555555')
ax.tick_params(axis='y', labelsize=14, colors='#555555')
ax.set_axisbelow(True)
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.legend(title='Payment Method', fontsize=12, title_fontsize=12, loc='upper right')
plt.show()

# ------------------------------------
# 4. Heatmap: correlation analysis
# Selecting relevant fields: churn, tenure, city_tier, rev_growth_yoy, rev_per_month
# Converting city_tier to numeric if it's not

data['city_tier'] = pd.to_numeric(data['city_tier'], errors='coerce')

corr_fields = ['churn', 'tenure', 'city_tier', 'rev_growth_yoy', 'rev_per_month']
# Some fields might be NaN; fill them with median if numeric
for field in corr_fields:
    if data[field].dtype != 'O':
        data[field].fillna(data[field].median(), inplace=True)

corr_matrix = data[corr_fields].corr()

plt.figure(figsize=(9,6))
ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5, linecolor='#333333')
plt.title('Correlation Heatmap', fontsize=20, pad=15, color='#222222')
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.show()

print('EDA complete.')

# Code block to inspect potential issues in the correlation fields
import numpy as np

# Let's inspect the unique values in each of the correlation fields
corr_fields = ['churn', 'tenure', 'city_tier', 'rev_growth_yoy', 'rev_per_month']

for field in corr_fields:
    if data[field].dtype == 'O':
        unique_vals = data[field].unique()
    else:
        unique_vals = data[field].unique()
    print('Unique values in ' + field + ':', unique_vals[:10])

# Let's check if any of these columns have non-numeric values that could cause the error
for field in corr_fields:
    try:
        numeric_col = pd.to_numeric(data[field], errors='coerce')
        n_non_numeric = numeric_col.isna().sum()
        print(field + ' - Non-numeric count:', n_non_numeric)
    except Exception as e:
        print(field + ' - Error:', e)

print('Inspection complete.')


# Clean the data by converting problematic columns to numeric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plotting style
plt.style.use('seaborn-v0_8')

# Convert problematic columns to numeric
data['tenure'] = pd.to_numeric(data['tenure'], errors='coerce')
data['rev_growth_yoy'] = pd.to_numeric(data['rev_growth_yoy'], errors='coerce')
data['rev_per_month'] = pd.to_numeric(data['rev_per_month'], errors='coerce')

# Fill NaN values with median for numeric columns
numeric_cols = ['tenure', 'city_tier', 'rev_growth_yoy', 'rev_per_month']
for col in numeric_cols:
    data[col].fillna(data[col].median(), inplace=True)

# Check if we have an 'HNI' segment in the data
print("Unique account segments:", data['account_segment'].unique())

# 1. Boxplot: revenue distributions across customer segments
plt.figure(figsize=(9,6))
ax = sns.boxplot(x='account_segment', y='rev_per_month', data=data, 
                palette=['#766CDB', '#DA847C', '#D9CC8B', '#7CD9A5', '#877877'])
ax.set_title('Revenue per Month by Account Segment', fontsize=20, pad=15, color='#222222')
ax.set_xlabel('Account Segment', fontsize=16, labelpad=10, color='#333333')
ax.set_ylabel('Revenue per Month', fontsize=16, labelpad=10, color='#333333')
ax.tick_params(axis='x', labelsize=14, colors='#555555')
ax.tick_params(axis='y', labelsize=14, colors='#555555')
ax.set_axisbelow(True)
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Scatter Plot: tenure vs churn with jitter to see density
plt.figure(figsize=(9,6))
ax = sns.stripplot(x='tenure', y='churn', data=data, jitter=0.3, 
                  size=8, palette=['#766CDB', '#DA847C'], alpha=0.6)
ax.set_title('Relationship Between Tenure and Churn', fontsize=20, pad=15, color='#222222')
ax.set_xlabel('Tenure (months)', fontsize=16, labelpad=10, color='#333333')
ax.set_ylabel('Churn (1=Yes, 0=No)', fontsize=16, labelpad=10, color='#333333')
ax.tick_params(axis='x', labelsize=14, colors='#555555')
ax.tick_params(axis='y', labelsize=14, colors='#555555')
ax.set_axisbelow(True)
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 3. Bar Graph: churn rates by city tier
churn_by_tier = data.groupby('city_tier')['churn'].mean().reset_index()

plt.figure(figsize=(9,6))
ax = sns.barplot(x='city_tier', y='churn', data=churn_by_tier, palette=['#766CDB', '#DA847C', '#D9CC8B'])
ax.set_title('Average Churn Rate by City Tier', fontsize=20, pad=15, color='#222222')
ax.set_xlabel('City Tier', fontsize=16, labelpad=10, color='#333333')
ax.set_ylabel('Average Churn Rate', fontsize=16, labelpad=10, color='#333333')
ax.tick_params(axis='x', labelsize=14, colors='#555555')
ax.tick_params(axis='y', labelsize=14, colors='#555555')
ax.set_axisbelow(True)
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 4. Bar Graph: churn rates by payment method
churn_by_payment = data.groupby('payment')['churn'].mean().reset_index()

plt.figure(figsize=(9,6))
ax = sns.barplot(x='payment', y='churn', data=churn_by_payment, palette=['#766CDB', '#DA847C', '#D9CC8B', '#7CD9A5'])
ax.set_title('Average Churn Rate by Payment Method', fontsize=20, pad=15, color='#222222')
ax.set_xlabel('Payment Method', fontsize=16, labelpad=10, color='#333333')
ax.set_ylabel('Average Churn Rate', fontsize=16, labelpad=10, color='#333333')
ax.tick_params(axis='x', labelsize=14, colors='#555555', rotation=45)
ax.tick_params(axis='y', labelsize=14, colors='#555555')
ax.set_axisbelow(True)
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.25)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 5. Correlation Heatmap
corr_fields = ['churn', 'tenure', 'city_tier', 'rev_growth_yoy', 'rev_per_month', 
               'service_score', 'account_user_count', 'cc_agent_score', 'complain_ly']
corr_matrix = data[corr_fields].corr()

plt.figure(figsize=(10,8))
ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=.5, linecolor='#333333', fmt='.2f')
plt.title('Correlation Heatmap of Key Variables', fontsize=20, pad=15, color='#222222')
plt.xticks(fontsize=12, color='#555555')
plt.yticks(fontsize=12, color='#555555')
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.show()

print('EDA complete with cleaned data.')



# Let's reattempt cleaning by converting numeric columns individually using pd.to_numeric with errors='coerce'

# List of numeric columns based on our inspection
numeric_cols = ['tenure', 'city_tier', 'rev_growth_yoy', 'rev_per_month', 'service_score', 'account_user_count', 'cc_agent_score', 'complain_ly']

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col].replace({'#': np.nan, '@': np.nan, '&': np.nan}), errors='coerce')
    # Fill NaN with median if column is not entirely NaN
    if data[col].notnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)
    else:
        print('Column ' + col + ' is completely NaN.')

print('Data cleaning reattempt complete.')

# Let's check data types now
print(data.dtypes)

# Now, let's plot the correlation heatmap again to ensure issues are fixed
corr_fields = ['churn', 'tenure', 'city_tier', 'rev_growth_yoy', 'rev_per_month', 'service_score',
               'account_user_count', 'cc_agent_score', 'complain_ly']
corr_matrix = data[corr_fields].corr()

plt.figure(figsize=(10,8))
ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=.5, linecolor='#333333', fmt='.2f')
plt.title('Correlation Heatmap of Key Variables', fontsize=20, pad=15, color='#222222')
plt.xticks(fontsize=12, color='#555555')
plt.yticks(fontsize=12, color='#555555')
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.show()

print('Reattempted cleaning and plotting complete.')



# Create Scatter Plot: Tenure vs Churn (with jitter) and Box Plot: Revenue per Month by Account Segment

# Filter rows for scatter plot for non-missing tenure and churn
scatter_df = df.dropna(subset=['tenure', 'churn'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15, wspace=0.3)

#########################
# Scatter Plot on ax1
#########################

# Apply jitter for the churn column
jitter_scatter = np.random.normal(0, 0.05, size=len(scatter_df))
churn_jittered = scatter_df['churn'] + jitter_scatter

scatter = ax1.scatter(scatter_df['tenure'], churn_jittered, 
                      alpha=0.6, c=scatter_df['churn'], cmap='coolwarm', 
                      s=50, edgecolor='#333333', linewidth=0.5)

# Axis and title for scatter plot
ax1.set_title('Relationship Between Tenure and Churn', fontsize=20, fontweight='semibold', color='#222222', pad=15)
ax1.set_xlabel('Tenure (months)', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax1.set_ylabel('Churn Status (with jitter)', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax1.tick_params(axis='both', labelsize=14, colors='#555555')
ax1.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
ax1.set_ylim(-0.2, 1.2)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Not Churned', 'Churned'])
for spine in ax1.spines.values():
    spine.set_color('#333333')
    spine.set_linewidth(0.8)

#########################
# Box Plot on ax2
#########################

# For the box plot, we'll use revenue per month vs account_segment
# Filter out rows with missing revenue values and account_segment
boxplot_df = df.dropna(subset=['rev_per_month', 'account_segment'])

sns.boxplot(x='account_segment', y='rev_per_month', data=boxplot_df, ax=ax2,
            palette=['#766CDB', '#DA847C', '#D9CC8B', '#7CD9A5', '#877877', '#52515E'])

# Axis and title for box plot
ax2.set_title('Revenue per Month by Account Segment', fontsize=20, fontweight='semibold', color='#222222', pad=15)
ax2.set_xlabel('Account Segment', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax2.set_ylabel('Revenue per Month', fontsize=16, fontweight='medium', color='#333333', labelpad=10)
ax2.tick_params(axis='both', labelsize=14, colors='#555555')
ax2.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
for spine in ax2.spines.values():
    spine.set_color('#333333')
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.show()

print('Scatter plot and box plot created successfully.')
