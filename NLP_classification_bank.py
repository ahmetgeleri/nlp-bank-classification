import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df=pd.read_csv('bank.csv')

stop_words=['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı',
            'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü',
            'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem',
            'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki',
            'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede',
            'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm',
            've', 'veya', 'ya', 'yani', 'lütfen']

msg=input("yapmak istediginiz islemi giriniz.")
msgdf=pd.DataFrame({"metin":msg, "kategori":0}, index=[42])

df=pd.concat([df, msgdf], ignore_index=True)

for word in stop_words:
    word=" "+word+" "
    df['metin']=df['metin'].str.replace(word, " ")

cv=CountVectorizer(max_features=75)

x=cv.fit_transform(df['metin']).toarray()
y=df['kategori']

prediction=x[-1].copy()

x=x[0:-1]
y=y[0:-1]

x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=21, train_size=0.7)

rf=RandomForestClassifier()
model=rf.fit(x_train, y_train)
score=model.score(x_test, y_test)
result=model.predict([prediction])
print("score:", score)
print("result:", result)
