#!/usr/bin/env python
# coding: utf-8

# # Import library for read the dataset

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Churn_Modelling.csv")


# In[3]:


df


# # Data Analysis

# In[4]:


df.describe()


# In[5]:


df.info()


# In[7]:


print(df.shape)


# In[8]:


df.count()


# In[9]:


df.isnull()


# In[10]:


df.isnull().sum()


# In[12]:


print(df.columns)


# In[13]:


single_column = df['CustomerId']

# Print the single column
print(single_column)


# In[15]:


single_column = df['Gender']

# Print the single column
print(single_column)


# In[17]:


# Select the desired columns
selected_columns = df[['CustomerId', 'Gender', 'CreditScore', 'Balance']]

# Show only the first 10 rows of the selected columns
selected_rows = selected_columns.head(10)

# Print the selected rows
print(selected_rows)


# In[18]:


filtered_data = df[(df['EstimatedSalary'] >= 96270.64) & (df['EstimatedSalary'] <= 101348.88)]

print(filtered_data)


# # Data Visualization

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


df.hist(bins=10, figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[24]:


# Define the colors based on conditions for 'Balance'
colors = ['red' if balance >= 83807.86 else 'yellow' for balance in df['Balance']]

# Scatter plot with different colors
plt.scatter(df['Gender'], df['Balance'], c=colors)
plt.xlabel('Gender')
plt.ylabel('Balance')
plt.title('Gender vs Balance')

# Show the plot
plt.show()


# In[25]:


# Define the colors based on conditions for 'Balance'
colors = ['red' if balance >= 83807.86 else 'yellow' for balance in df['EstimatedSalary']]

# Scatter plot with different colors
plt.scatter(df['Gender'], df['EstimatedSalary'], c=colors)
plt.xlabel('Gender')
plt.ylabel('EstimatedSalary')
plt.title('Gender vs EstimatedSalary')

# Show the plot
plt.show()


# In[26]:


# Define the colors based on conditions for 'Balance'
colors = ['red' if balance >= 83807.86 else 'yellow' for balance in df['EstimatedSalary']]

# Scatter plot with different colors
plt.scatter(df['Balance'], df['EstimatedSalary'], c=colors)
plt.xlabel('Balance')
plt.ylabel('EstimatedSalary')
plt.title('Balance vs EstimatedSalary')

# Show the plot
plt.show()


# In[27]:


# Select the columns for correlation
columns = ['CustomerId', 'CreditScore', 'Balance', 'EstimatedSalary']

# Calculate the coefficient matrix
correlation_matrix = df[columns].corr()

# Display the coefficient matrix
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[30]:


sns.scatterplot(x='Balance', y='EstimatedSalary',
                hue='Balance', data=df, )
 
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
 
plt.show()


# # Machine Learning Models

# In[68]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load and preprocess the data
# Assuming you have the data in a CSV file named 'customer_data.csv'
df = pd.read_csv("Churn_Modelling.csv")
# Handle missing values if any
#data = data.dropna()

# Separate features and target variable
X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = df['Exited']

# Encode categorical variables if any
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Model Training
# 1. Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# 2. Random Forests
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# 3. Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Step 3: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

print("Logistic Regression Performance:")
lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluate_model(lr_model, X_test, y_test)
print(f"Accuracy: {lr_accuracy:.2f}, Precision: {lr_precision:.2f}, Recall: {lr_recall:.2f}, F1-Score: {lr_f1:.2f}")

print("\nRandom Forest Performance:")
rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(rf_model, X_test, y_test)
print(f"Accuracy: {rf_accuracy:.2f}, Precision: {rf_precision:.2f}, Recall: {rf_recall:.2f}, F1-Score: {rf_f1:.2f}")

print("\nGradient Boosting Performance:")
gb_accuracy, gb_precision, gb_recall, gb_f1 = evaluate_model(gb_model, X_test, y_test)
print(f"Accuracy: {gb_accuracy:.2f}, Precision: {gb_precision:.2f}, Recall: {gb_recall:.2f}, F1-Score: {gb_f1:.2f}")


# In[67]:


# Encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Step 2: Model Training
gb_model = GradientBoostingClassifier()
gb_model.fit(X_scaled, y)

# Step 3: Make Predictions (using the best-performing model)
new_data = pd.read_csv('Churn_Modelling.csv')  # Replace 'Churn_Modelling.csv' with the new data file name
new_data = new_data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])  # Remove irrelevant columns

# Encode categorical variables for new_data if any
new_data_encoded = pd.get_dummies(new_data, drop_first=True)

# Ensure both datasets have the same columns
missing_cols = set(X_encoded.columns) - set(new_data_encoded.columns)
for col in missing_cols:
    new_data_encoded[col] = 0

# Reorder columns to match the order in X_encoded
new_data_encoded = new_data_encoded[X_encoded.columns]

# Standardize new_data_encoded using the scaler fit on the training data
new_data_scaled = scaler.transform(new_data_encoded)

# Make predictions
predictions = gb_model.predict(new_data_scaled)

# 'predictions' contains the churn predictions for the new customers.
print("Gradient Boosting Predictions:")
print(predictions)

# Make predictions using Logistic Regression
lr_predictions = lr_model.predict(new_data_scaled)

# Make predictions using Random Forest
rf_predictions = rf_model.predict(new_data_scaled)

# Output churn predictions for the new customers using Logistic Regression
print("\nLogistic Regression Predictions:")
print(lr_predictions)

# Output churn predictions for the new customers using Random Forest
print("\nRandom Forest Predictions:")
print(rf_predictions)


# In[ ]:





# In[ ]:




