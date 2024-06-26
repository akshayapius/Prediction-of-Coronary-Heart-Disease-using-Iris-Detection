import pandas as pd
import cv2
from sklearn import svm
import pickle

df=pd.read_csv("C:/Users/anandhu/Downloads/baksha/tabular-actgan-655da1fe176d0daad2b1a1e1-data_preview.csv")
kf=df.drop(["Iris Name"],axis=1)
kf.drop_duplicates()
X=kf.drop(['Output Target'],axis=1)
y=kf['Output Target']
y=y.astype('int')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
sc=svm.SVC()
sc.fit(X_train,y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(sc,f)
