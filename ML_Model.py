import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv",skiprows=1,skip_blank_lines=False)

#data_cleaning
na_index = df[df.isnull().any(axis=1)].index
alpha_df = df[df.iloc[:,0].str.isalpha()].index[0]
df = df.drop(index=na_index)
df = df.drop(index=alpha_df)

df.duplicated().all()
df.drop_duplicates(subset=None, keep = 'first', inplace=True)

df.columns = df.columns.str.strip().str.replace(' ', '_')

df['Classes'] = df['Classes'].str.strip().map({'fire': 1, 'not fire': 0})

correlation_matrix = df.corr()
correlation_matrix.style.background_gradient(cmap='coolwarm',axis=None)

#data_ploting
correlation_matrix = df.corr()
plt.imshow(correlation_matrix,cmap='coolwarm', interpolation='none')
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
correlation_matrix.style.background_gradient(cmap='coolwarm')
plt.show()

df = df.drop(['year','month'],axis=1)

#train_test_Split 
X = df.iloc[:,:-1]
y = df['Classes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model_training
model = LogisticRegression()
model.fit(X_train, y_train)

#model_prediction
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

