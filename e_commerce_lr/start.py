import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("data.csv")

# Uncomment these lines to see the dataframe structure
# print(df.head())
# print(df.info())

# Exploratory Data Analysis (EDA)
# sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)
# sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)

# Corrected pairplot
# sns.pairplot(df[["Time on Website", "Yearly Amount Spent"]], kind="scatter", plot_kws={"alpha": 0.4})
# plt.show()

# sns.lmplot(x="Length of Membership", 
#            y="Yearly Amount Spent", 
#            data=df, 
#            scatter_kws={'alpha': 0.3})
# plt.show()

# Avg. Session Length , Time on App, Time on Website, Length of Membership
# Splitting the data using sklearn
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
Y = df["Yearly Amount Spent"]

# Training the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
lm = LinearRegression()
lm.fit(X_train, Y_train)

# Testing the model
predictions = lm.predict(X_test)

# Plotting the predictions
sns.scatterplot(x=predictions, y=Y_test)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Predictions vs Actual Values')
plt.show()
print("mean absolute error",mean_absolute_error(Y_test,predictions))
