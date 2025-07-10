import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


df =pd.read_csv("spam.csv")

df =df[["label", "text"]]
print(df.shape)

count= df['label'].value_counts()
df['label_num']= df['label'].map({'ham':0,"spam":1})

x =df["text"]
y= df["label_num"]

x_train, x_test,y_train, y_test =train_test_split(x,y, test_size=0.2, random_state=42)

#for convertig the words into the number
vectorizer = CountVectorizer()
x_vac_train = vectorizer.fit_transform(x_train)
x_vac_test = vectorizer.transform(x_test)

#for counting the words
model =MultinomialNB()
model.fit(x_vac_train,y_train)

y_pred=model.predict(x_vac_test)
accuracy = accuracy_score(y_test, y_pred)

# plt.bar(count.index, count.values)
# plt.title("its a ham and spam project")
# plt.show()
# plt.savefig("ham_spam.png")
print("The accuracy is", round(accuracy * 100, 2), "%")


while True:
   print("Enter your message")

   user_input= input("Your message:  ")

   user_vector = vectorizer.transform([user_input])
   prediction = model.predict(user_vector)

   if prediction [0]==1:
       print("Its a spam message")

   else: 
       print("Its a ham")
       print("Wanna try again")

   try_again = input("y/n =")
   if try_again!='y':
       print("Good luck see you next time")
       break
