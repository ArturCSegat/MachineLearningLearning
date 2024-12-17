import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as kn
import sklearn.discriminant_analysis as da
import sklearn.linear_model as ln
import sklearn.metrics as metrics

df = pd.read_csv("iris.data")

label_dict = {
    "Iris-setosa": 1, 
    "Iris-versicolor": 2,
    "Iris-virginica":  3,
}

rev_label_dict = {
        1: "Iris-setosa", 
        2:"Iris-versicolor",
        3: "Iris-virginica",
}
_, axis = plt.subplots(3, 2)

axis[0, 0].set_title("SEPAL real")
axis[0, 0].set_xlabel("sepal lenght")
axis[0, 0].set_ylabel("sepal width")
axis[0, 0].scatter(df[df.name == "Iris-setosa"]["sepal_len"], df[df.name == "Iris-setosa"]["sepal_width"], color="darkgreen")
axis[0, 0].scatter(df[df.name == "Iris-versicolor"]["sepal_len"], df[df.name == "Iris-versicolor"]["sepal_width"], color="darkred")
axis[0, 0].scatter(df[df.name == "Iris-virginica"]["sepal_len"], df[df.name == "Iris-virginica"]["sepal_width"], color="darkblue")

axis[0, 1].set_title("PETAL real")
axis[0, 1].set_xlabel("petal lenght")
axis[0, 1].set_ylabel("petal width")
axis[0, 1].scatter(df[df.name == "Iris-setosa"]["petal_len"], df[df.name == "Iris-setosa"]["petal_width"], color="darkgreen")
axis[0, 1].scatter(df[df.name == "Iris-versicolor"]["petal_len"], df[df.name == "Iris-versicolor"]["petal_width"], color="darkred")
axis[0, 1].scatter(df[df.name == "Iris-virginica"]["petal_len"], df[df.name == "Iris-virginica"]["petal_width"], color="darkblue")



pred = df.drop(columns=["name"])
label = [label_dict[n] for n  in df["name"]]

lda = da.LinearDiscriminantAnalysis().fit(pred, label)
knn = kn.KNeighborsClassifier().fit(pred, label)

labeli = lda.predict(pred)
labeli2 = knn.predict(pred)

df["prediction_lda"] = labeli
df["prediction_knn"] = labeli2

axis[1, 0].set_title("SEPAL prediction_lda")
axis[1, 0].set_xlabel("sepal lenght")
axis[1, 0].set_ylabel("sepal width")
axis[1, 0].scatter(df[df.prediction_lda == 1]["sepal_len"], df[df.prediction_lda == 1]["sepal_width"], color="darkgreen")
axis[1, 0].scatter(df[df.prediction_lda == 2]["sepal_len"], df[df.prediction_lda == 2]["sepal_width"], color="darkred")
axis[1, 0].scatter(df[df.prediction_lda == 3]["sepal_len"], df[df.prediction_lda == 3]["sepal_width"], color="darkblue")

axis[1, 1].set_title("PETAL prediction_lda")
axis[1, 1].set_xlabel("petal lenght")
axis[1, 1].set_ylabel("petal width")
axis[1, 1].scatter(df[df.prediction_lda == 1]["petal_len"], df[df.prediction_lda == 1]["petal_width"], color="darkgreen")
axis[1, 1].scatter(df[df.prediction_lda == 2]["petal_len"], df[df.prediction_lda == 2]["petal_width"], color="darkred")
axis[1, 1].scatter(df[df.prediction_lda == 3]["petal_len"], df[df.prediction_lda == 3]["petal_width"], color="darkblue")


axis[2, 0].set_title("SEPAL prediction_knn")
axis[2, 0].set_xlabel("sepal lenght")
axis[2, 0].set_ylabel("sepal width")
axis[2, 0].scatter(df[df.prediction_knn == 1]["sepal_len"], df[df.prediction_knn == 1]["sepal_width"], color="darkgreen")
axis[2, 0].scatter(df[df.prediction_knn == 2]["sepal_len"], df[df.prediction_knn == 2]["sepal_width"], color="darkred")
axis[2, 0].scatter(df[df.prediction_knn == 3]["sepal_len"], df[df.prediction_knn == 3]["sepal_width"], color="darkblue")

axis[2, 1].set_title("PETAL prediction_knn")
axis[2, 1].set_xlabel("petal lenght")
axis[2, 1].set_ylabel("petal width")
axis[2, 1].scatter(df[df.prediction_knn == 1]["petal_len"], df[df.prediction_knn == 1]["petal_width"], color="darkgreen")
axis[2, 1].scatter(df[df.prediction_knn == 2]["petal_len"], df[df.prediction_knn == 2]["petal_width"], color="darkred")
axis[2, 1].scatter(df[df.prediction_knn == 3]["petal_len"], df[df.prediction_knn == 3]["petal_width"], color="darkblue")

lda_pre = metrics.accuracy_score(label, labeli)
knn_pre = metrics.accuracy_score(label, labeli2)
print(f"lda accuracy: {lda_pre}")
print(f"knn accuracy: {knn_pre}")

plt.show()
