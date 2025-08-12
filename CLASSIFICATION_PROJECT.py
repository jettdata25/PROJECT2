import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# Load your CSV file
df = pd.read_csv('Competitor Listings for Jett-2025-08-06-10-16-11.csv')

#DATA CLEANING AND EXPLORATION
# Check for missing values
missing_summary = df.isnull().sum()
print("Missing values per column:")
print(missing_summary[missing_summary > 0])

# List of columns to drop
cols_to_drop = [
    'Unnamed: 25',
    'Missing Fields List',
    'LDC Places',
    'Agency 2',
    'Agent 4',
    'Agent 3',
    'Agent 2'
]

# Drop them from the DataFrame
df_cleaned = df.drop(columns=cols_to_drop)

# --- Basic Info ---
print("\n🔍 Dataset Info:")
df.info()

# --- Summary Statistics ---
print("\n📈 Summary Statistics (Numerical Columns):")
print(df.describe())

# --- Data Types ---
print("\n🧬 Data Types:")
print(df.dtypes)

# --- Missing Values Overview ---
print("\n🚨 Missing Values:")
missing_summary = df.isnull().sum()
missing_percent = (missing_summary / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_summary,
    'Missing %': missing_percent.round(2)
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing %', ascending=False))

# --- Unique Values per Column ---
print("\n🔢 Unique Values per Column:")
unique_counts = df.nunique().sort_values(ascending=False)
print(unique_counts)

# Ensure date columns are in datetime format
df['Advert Date'] = pd.to_datetime(df['Advert Date'], errors='coerce', dayfirst=True)
df['Sale Date'] = pd.to_datetime(df['Sale Date'], errors='coerce', dayfirst=True)

# Calculate time on market in days
df['Time_on_Market'] = (df['Sale Date'] - df['Advert Date']).dt.days

# Optional: Check distribution
print("\n📊 Time on Market Summary:")
print(df['Time_on_Market'].describe())

sns.histplot(df['Time_on_Market'], bins=30, kde=True)
plt.title('Distribution of Time on Market')
plt.xlabel('Days')
plt.ylabel('Count')
plt.show()

df['Sold_Quickly'] = df['Time_on_Market'] <= 90

# Define target
target = 'Sold_Quickly'

# Example feature selection (customize as needed)
features = [
    'Rent', 'Yield', 'Sale Price', 'Building Size', 'Land Size',
    'Agency 1', 'Main Tenant', 'Sale Method'
]

# Subset the data
X = df[features]
y = df[target]

#CLASSIFICATION MODELING
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Identify column types
numeric_features = ['Rent', 'Yield', 'Sale Price', 'Building Size', 'Land Size']
categorical_features = ['Agency 1', 'Main Tenant', 'Sale Method']

# Clean currency columns
currency_columns = ['Sale Price', 'Rent', 'Yield']

for col in currency_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r'[\$,]', '', regex=True)  # Remove $ and commas
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float
  # Convert to float

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

logreg_pipeline.fit(X_train, y_train)
y_pred_logreg = logreg_pipeline.predict(X_test)

print("\n📊 Logistic Regression Report:")
print(classification_report(y_test, y_pred_logreg))

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

print("\n🌲 Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# For Random Forest (or any model)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1]))