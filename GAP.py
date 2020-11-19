import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Admission_Predict_Ver1.1.csv",sep = ",")
df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})

# reading the dataset
df = pd.read_csv("Admission_Predict_Ver1.1.csv",sep = ",")

# it may be needed in the future.
serialNo = df["Serial No."].values

df.drop(["Serial No."],axis=1,inplace = True)

df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})

y = df["Chance of Admit"].values
x = df.drop(["Chance of Admit"],axis=1)

# separating train (80%) and test (%20) sets
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

# normalization
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_head_lr = lr.predict(x_test)

print("real value(From Dataset)(Chance of Admit): " + str(y_test[0]))
'''print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(lr.predict(x_test.iloc[[2],:])))'''
print(y_test)
from sklearn.metrics import r2_score

y_head_lr_train = lr.predict(x_train)
x0=input("Enter Student Name:")
x1=int(input("Enter GRE Score:"))
x2=int(input("Enter TOEFL Score:"))
x3=int(input("Enter University Rating:"))
x4=int(input("Enter SOP Score:"))
x5=int(input("Enter LOR Score:"))
x6=float(input("Enter CGPA:"))
x7=int(input("Enter Research(1=Yes | 0=No):"))

data=[[x1,x2,x3,x4,x5,x6,x7]]
data=scalerX.transform(data)
print(x0)
print("Predicted Value(Best Accuracy):"+str(lr.predict(data)))
print("Accuracy using Linear Regresiion(r_square score): ", r2_score(y_test,y_head_lr)*100)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfr.fit(x_train,y_train)
y_head_rfr = rfr.predict(x_test) 


print("Accuracy using RandomForestRegressor Regresiion(r_square score): ", r2_score(y_test,y_head_rfr)*100)
y_head_rf_train = rfr.predict(x_train)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(x_train,y_train)
y_head_dtr = dtr.predict(x_test)

print("Accuracy using DecisionTreeRegressor(r_square score): ", r2_score(y_test,y_head_dtr)*100)
y_head_dtr_train = dtr.predict(x_train)

y = np.array([r2_score(y_test,y_head_lr),r2_score(y_test,y_head_rfr),r2_score(y_test,y_head_dtr)])
x = ["LinearRegression","RandomForestReg.","DecisionTreeReg."]
plt.bar(x,y)
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("r2_score")
plt.show()



fig = sns.regplot(x="GRE Score", y="Chance of Admit", data=df)
plt.title("GRE Score vs Chance of Admit")
plt.show()

fig = sns.regplot(x="TOEFL Score", y="Chance of Admit", data=df)
plt.title("TOEFL Score vs Chance of Admit")
plt.show()

fig = sns.regplot(x="University Rating", y="Chance of Admit", data=df)
plt.title("University Rating vs Chance of Admit")
plt.show()

fig = sns.regplot(x="SOP", y="Chance of Admit", data=df)
plt.title("SOP vs Chance of Admit")
plt.show()

fig = sns.regplot(x="LOR ", y="Chance of Admit", data=df)
plt.title("LOR vs Chance of Admit")
plt.show()

fig = sns.regplot(x="CGPA", y="Chance of Admit", data=df)
plt.title("CGPA vs Chance of Admit")
plt.show()



fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()

print("Not Having Research:",len(df[df.Research == 0]))
print("Having Research:",len(df[df.Research == 1]))
y = np.array([len(df[df.Research == 0]),len(df[df.Research == 1])])
x = ["Not Having Research","Having Research"]
plt.bar(x,y)
plt.title("Research Experience")
plt.xlabel("Canditates")
plt.ylabel("Frequency")
plt.show()

y = np.array([df["TOEFL Score"].min(),df["TOEFL Score"].mean(),df["TOEFL Score"].max()])
x = ["Worst","Average","Best"]
plt.bar(x,y)
plt.title("TOEFL Scores")
plt.xlabel("Level")
plt.ylabel("TOEFL Score")
plt.show()

df["GRE Score"].plot(kind = 'hist',bins = 200,figsize = (6,6))
plt.title("GRE Scores")
plt.xlabel("GRE Score")
plt.ylabel("Frequency")
plt.show()

plt.scatter(df["University Rating"],df.CGPA)
plt.title("CGPA Scores for University Ratings")
plt.xlabel("University Rating")
plt.ylabel("CGPA")
plt.show()

plt.scatter(df["GRE Score"],df.CGPA)
plt.title("CGPA for GRE Scores")
plt.xlabel("GRE Score")
plt.ylabel("CGPA")
plt.show()

df[df.CGPA >= 8.5].plot(kind='scatter', x='GRE Score', y='TOEFL Score',color="red")
plt.xlabel("GRE Score")
plt.ylabel("TOEFL SCORE")
plt.title("CGPA>=8.5")
plt.grid(True)
plt.show()

s = df[df["Chance of Admit"] >= 0.75]["University Rating"].value_counts().head(5)
plt.title("University Ratings of Candidates with an 75% acceptance chance")
s.plot(kind='bar',figsize=(20, 10))
plt.xlabel("University Rating")
plt.ylabel("Candidates")
plt.show()

plt.scatter(df["CGPA"],df.SOP)
plt.xlabel("CGPA")
plt.ylabel("SOP")
plt.title("SOP for CGPA")
plt.show()

plt.scatter(df["GRE Score"],df["SOP"])
plt.xlabel("GRE Score")
plt.ylabel("SOP")
plt.title("SOP for GRE Score")
plt.show()

df["Chance of Admit"].plot(kind = 'hist',bins = 200,figsize = (6,6))
plt.title("Chance of Admit")
plt.xlabel("Chance of Admit")
plt.ylabel("Frequency")
plt.show()
