import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



#Retrieve the Boston Housing Project
from sklearn.datasets import load_boston

#Put data in variable 'btown_data'
btown_data = load_boston()

#Understand what the data contains
print(btown_data.keys())
btown_data.DESCR #gives more of a description of the features

#Putting the data into a df
boston = pd.DataFrame(btown_data.data, columns=btown_data.feature_names)
boston.head()

boston['MEDV'] = btown_data.target

# Plot Distribution of MEDV -> this is the target variable
# MEDV: Median value of owner-occupied homes in $1000s

#I am going to compare 30 bins with 10 bins just out of curiosity
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=10)
plt.show()

#Get more info on the distribution of the data
boston.describe()

#create a correlation matrix to see relationship between variables
CM = boston.corr().round(2)

# Create the heatmap with the chosen color palette
sns.heatmap(data=CM, annot=True, cmap='vlag')

plt.show(block=True)

# When looking at the plot, the highest correlated variable to MEDV is RM
# RM has a value of 0.7 and the next closest variables are ZN and B with 0.36 and 0.33, respectfully
# When looking at the plot, the lowest correlated variable to MEDV is LSTAT (-0.74)


# Plot of positive and negative variables to MEDV to view relatonhip

# Define the variables and target
vars = ['RM', 'LSTAT']
target = boston['MEDV']

# Create subplots
fig, axes = plt.subplots(1, len(vars), figsize=(12, 5))

# Iterate over variables
for i, col in enumerate(vars):
    sns.regplot(x=boston[col], y=target, ax=axes[i], scatter_kws={'alpha':0.5})
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('MEDV')

plt.tight_layout()
plt.show(block = True)

#Using the data to implement a simple linear regression
# Create subplots
fig, axes = plt.subplots(1, len(vars), figsize=(12, 5))

# Iterate over variables
for i, col in enumerate(vars):
    # Reshape the data to fit the linear regression model
    x = boston[col].values.reshape(-1, 1)
    y = target.values

    # Fit the linear regression model
    lin_model = LinearRegression()
    lin_model.fit(x, y)

    # Plot scatter plot
    sns.scatterplot(x=boston[col], y=target, ax=axes[i], alpha=0.5)
    
    # Plot regression line
    sns.lineplot(x=boston[col], y=lin_model.predict(x), color='red', ax=axes[i])
    
    # Set plot labels and title
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('MEDV')

    # Display equation of the line
    axes[i].annotate(f'y = {lin_model.coef_[0]:.2f}x + {lin_model.intercept_:.2f}', 
                     xy=(0.5, 0.9), xycoords='axes fraction', ha='center')

plt.tight_layout()
plt.show(block = True)


# Select predictor and target variables
X = boston[['RM']]  # Predictor variable 'RM'
y = boston['MEDV']   # Target variable 'MEDV'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Make predictions on the test set
y_test_predict = lin_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)

# Display the model's coefficients and performance metrics
print("Coefficients:", lin_model.coef_)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Create a DataFrame to display the results
results = pd.DataFrame({
    'Metric': ['Coefficient', 'Mean Squared Error (MSE)', 'R-squared (R2)'],
    'Value': [lin_model.coef_[0], mse, r2]
})

# Display the results
print(results)
