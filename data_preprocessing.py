import pandas as pd
import os

# Define file paths
file_paths = {
    "holidays_events": "../data/holidays_events.csv",
    "oil": "../data/oil.csv",
    "stores": "../data/stores.csv",
    "train": "../data/train.csv",
    "test": "../data/test.csv",
    "transactions": "../data/transactions.csv",
}

# Load data
holidays_events = pd.read_csv(file_paths["holidays_events"])
oil = pd.read_csv(file_paths["oil"])
stores = pd.read_csv(file_paths["stores"])
train = pd.read_csv(file_paths["train"])
test = pd.read_csv(file_paths["test"])
transactions = pd.read_csv(file_paths["transactions"])

# Convert date columns to datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
oil['date'] = pd.to_datetime(oil['date'])
holidays_events['date'] = pd.to_datetime(holidays_events['date'])
transactions['date'] = pd.to_datetime(transactions['date'])

# Rename columns to avoid conflicts
holidays_events.rename(columns={'type': 'holiday_type'}, inplace=True)
stores.rename(columns={'type': 'store_type'}, inplace=True)

# Print column names for debugging
print("Train columns:", train.columns)
print("Test columns:", test.columns)
print("Stores columns:", stores.columns)
print("Transactions columns:", transactions.columns)
print("Holidays events columns:", holidays_events.columns)
print("Oil columns:", oil.columns)

# Merge train data with stores data
train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# Merge train data with oil data
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')

# Merge train data with transactions data
train = train.merge(transactions, on=['date', 'store_nbr'], how='left')

# Merge train and test data with holidays_events data
train = train.merge(holidays_events, on='date', how='left')
test = test.merge(holidays_events, on='date', how='left')

# Fill missing values
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# Extract date features
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['dayofweek'] = train['date'].dt.dayofweek

test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['dayofweek'] = test['date'].dt.dayofweek

# Drop unnecessary columns
train.drop(columns=['date', 'description', 'locale_name', 'transferred'], inplace=True, errors='ignore')
test.drop(columns=['date', 'description', 'locale_name', 'transferred'], inplace=True, errors='ignore')

# Ensure categorical columns are properly encoded
categorical_columns = ['store_type', 'locale', 'cluster', 'holiday_type', 'family', 'city', 'state']

# Check if necessary columns exist after merging
missing_columns_train = [col for col in categorical_columns if col not in train.columns]
missing_columns_test = [col for col in categorical_columns if col not in test.columns]

if missing_columns_train or missing_columns_test:
    print(f"Missing columns in train: {missing_columns_train}")
    print(f"Missing columns in test: {missing_columns_test}")
else:
    # One-hot encoding for categorical variables
    train = pd.get_dummies(train, columns=categorical_columns)
    test = pd.get_dummies(test, columns=categorical_columns)

    # Ensure train and test have the same columns
    test = test.reindex(columns=train.columns, fill_value=0)

    # Save preprocessed data
    train.to_csv("../data/preprocessed_train.csv", index=False)
    test.to_csv("../data/preprocessed_test.csv", index=False)
    print("Preprocessed train data saved to '../data/preprocessed_train.csv'")
    print("Preprocessed test data saved to '../data/preprocessed_test.csv'")
