import pandas as pd
import plotly.express as px

df = pd.read_csv("data3.csv")

gre_score = df["GRE Score"].tolist()
toefl_score = df["TOEFL Score"].tolist()

print(len(gre_score))

fig = px.scatter(x=gre_score, y=toefl_score)
# fig.show()


import plotly.graph_objects as go

gre_score = df["GRE Score"].tolist()
toefl_score = df["TOEFL Score"].tolist()

results = df["Chance of admit"].tolist()
colors=[]
for data in results:
  if data == 1:
    colors.append("green")
  else:
    colors.append("red")


fig = go.Figure(data=go.Scatter(
    x=gre_score,
    y=toefl_score,
    mode='markers',
    marker=dict(color=colors)
))
# fig.show()

#Taking together Age and Salary of the person
score = df[["GRE Score","TOEFL Score"]]

#Purchases made
results = df["Chance of admit"]

from sklearn.model_selection import train_test_split 

score_train, score_test, results_train, results_test = train_test_split(score, results, test_size = 0.25, random_state = 0)
print(score_test)
print(score_train[0:10])


from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 

score_train = sc_x.fit_transform(score_train)  
score_test = sc_x.transform(score_test) 
  
print(score_train[0:10])

from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(score_train, results_train)

results_pred = classifier.predict(score_test)

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(results_test, results_pred)) 


# Testing the data
user_hours_studied = int(input("Enter GRE Score of the customer -> "))
user_hours_slept = int(input("Enter the TOEFL score of the customer -> "))

user_test = sc_x.transform([[ user_hours_studied,user_hours_slept]])

user_results_pred = classifier.predict(user_test)

if user_results_pred[0] == 1:
  print("This customer may get admited!")
else:
  print("This customer may not get admited!")