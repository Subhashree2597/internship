#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For ignoring warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd

zomato = pd.read_csv(r"C:\Users\subha\OneDrive\Desktop\zomato.csv", encoding='latin-1')
country = pd.read_excel(r"C:\Users\subha\OneDrive\Desktop\Country-Code.xlsx")

print(zomato.head())
print(country.head())


# In[3]:


# Merge zomato and country data on 'Country Code'
df = pd.merge(zomato, country, on='Country Code', how='left')

# View merged data
df.head()


# In[4]:


# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop rows with missing 'Cuisines' or 'Average Cost for two'
df.dropna(subset=['Cuisines', 'Average Cost for two'], inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Check missing values
df.isnull().sum()


# In[5]:


# Check unique values in Currency column
print(df['Currency'].unique())


# In[6]:


# Currency conversion rates to INR (example rates)
currency_rates = {
    'Botswana Pula(P)': 6.3,
    'Brazilian Real(R$)': 17.0,
    'Dollar($)': 83.0,
    'Emirati Diram(AED)': 22.6,
    'Indian Rupees(Rs.)': 1,
    'Indonesian Rupiah(IDR)': 0.0052,
    'NewZealand($)': 51.0,
    'Pounds(£)': 104.0,
    'Qatari Rial(QR)': 22.8,
    'Rand(R)': 4.4,
    'Sri Lankan Rupee(LKR)': 0.27,
    'Turkish Lira(TL)': 2.6
}


# In[7]:


df['Cost_INR'] = df.apply(lambda row: row['Average Cost for two'] * currency_rates.get(row['Currency'], 1), axis=1)


# In[8]:


df[['Average Cost for two', 'Currency', 'Cost_INR']].head()


# ### Basic Analysis & Visualizations
# 1. Top 10 Countries by Number of Restaurants

# In[9]:


top_countries = df['Country'].value_counts().head(10)
top_countries.plot(kind='bar', title='Top 10 Countries by Restaurant Count', figsize=(10,5))
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.show()


# ###  2.Online Delivery Availablity

# In[10]:


sns.countplot(data=df, x='Has Online delivery')
plt.title("Online Delivery Available or Not")
plt.show()


# ### 3. Top Cuisines

# In[11]:


top_cuisines = df['Cuisines'].value_counts().head(10)
top_cuisines.plot(kind='bar', title='Top 10 Cuisines', figsize=(10,5))
plt.xticks(rotation=45)
plt.show()


# ### 4.Avarage Cost Distribution

# In[12]:


plt.hist(df['Average Cost for two'], bins=30, color='skyblue')
plt.title('Distribution of Average Cost for Two')
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.show()


# ### Prediction using Machine Learning

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select features & target
features = df[['Price range', 'Aggregate rating', 'Votes']]
target = df['Average Cost for two']

# Clean missing values
data = pd.concat([features, target], axis=1).dropna()
X = data[['Price range', 'Aggregate rating', 'Votes']]
y = data['Average Cost for two']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# In[14]:


# Save the cleaned & merged dataset
df.to_csv("zomato_for_powerbi.csv", index=False)
print("File saved successfully ✅")

