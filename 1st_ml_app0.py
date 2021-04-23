from sklearn import datasets
import streamlit as st
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("方希 First Machine Learning App")
st.write("""
# Explore Different Classifier
Compare the performance of different classifiers
""")

dataset_name = st.sidebar.selectbox("Select Dataset",("Iris", "Breast Cancer", "Wine Dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier",("Random Forest", "KNN", "SVM"))

st.write ("Using Classifier ", classifier_name, "on Dataset", dataset_name)

def get_dataset(dataset_name):
	if dataset_name == "Iris":
	   data = datasets.load_iris()
	elif dataset_name == "Breast Cancer":
	   data = datasets.load_breast_cancer()
	else:
	   data = datasets.load_wine()
	x = data.data
	y = data.target
	return x, y

x, y = get_dataset(dataset_name)
st.write ("number of data", x.shape[0], ", number of features of each data", x.shape[1], ", number of classes", len(np.unique(y)) )

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
      K = st.sidebar.slider("K", 1, 15)
      params["K"] = K
    elif clf_name == "SVM":
      C = st.sidebar.slider("C", 0.01, 10.0)
      params["C"] = C
    else:
      max_depth = st.sidebar.slider("max_depth", 2, 15)
      n_trees = st.sidebar.slider("n_trees", 1, 100)
      params["max_depth"] = max_depth
      params["n_trees"] = n_trees
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_trees"],max_depth=params["max_depth"],random_state=42)
    return clf

clf = get_classifier(classifier_name,params)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test,y_pred)*100
acc = round(acc, 2)

st.write(f"calssifier = {classifier_name}")
st.write(f"accuracy = {acc} %")

#plot

pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principla Component 2")
plt.colorbar()

#plt.show()
st.pyplot(fig)


