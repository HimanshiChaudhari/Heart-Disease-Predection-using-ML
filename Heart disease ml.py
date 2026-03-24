import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("X:\\public\\Desktop\\predective\\heart_disease_uci.csv")
print("First 5 rows:\n", data.head(),"\n")

print("Missing values before cleaning:\n", data.isnull().sum(),"\n")
# droping id column
data = data.drop("id", axis=1)
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
#filling numerical columns
numeric_cols = data.select_dtypes(include ='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

print(data)
#filling categorical columns
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    data[col]=data[col].fillna(data[col].mode()[0])
print(data)

#Convert text (categorical) data into numbers
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
print(data)

#Outlier detection
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
outliers = ((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).sum()
print("Outliers found in each column:\n", outliers, "\n")

#Split into training and testing sets
x = data.drop(data.columns[-1], axis=1)  # features
y = data[data.columns[-1]]              # target (last column)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print("Data split into training and testing sets.\n")

print("Final dataset shape:", data.shape)
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

counts = data.groupby(['gender', 'target']).size().unstack()
counts.plot(kind='bar',
            figsize=(8, 5),
            color=['royalblue', 'goldenrod'])
plt.title("Heart Disease Presence by Sex")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.legend(title="Heart Disease", labels=["No Disease (0)", "Disease (1)"])
plt.xticks(rotation=0)
plt.show()


cp_counts = data.groupby(['chest pain type', 'target']).size().unstack(fill_value=0)
z = np.arange(len(cp_counts.index))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(z - width/2, cp_counts[0], width, label='No Disease (0)', color='royalblue')
plt.bar(z + width/2, cp_counts[1], width, label='Disease (1)', color='goldenrod')
plt.title("Heart Disease Presence by Chest Pain Type (CP)")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.xticks(z, cp_counts.index)
plt.legend(title="Heart Disease")
plt.show()


#correlation
corr_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()





#Logistic regression
model= LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Model Accuracy :", round(accuracy_score(y_test,y_pred),3))
cm=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n", cm)


plt.figure(figsize=(7,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease (0)", "Disease (1)"],
    yticklabels=["No Disease (0)", "Disease (1)"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Logistic Regression (Accuracy: 80.00%)")
plt.show()


#KNN model

knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(x_train,y_train)

y_test_pred = knn.predict(x_test)
print("\nPredicted labels on test set:")
print(y_test_pred)

cm= confusion_matrix(y_test,y_test_pred,
                     labels=[0,1])
print("\nConfusion Matrix (rows = Actual, cols = Predicted):")
print(pd.DataFrame(cm,
                   index=["Actual_NoDisease", "Actual_Disease"],
                   columns=["Pred_NoDisease", "Pred_Disease"]))
print("accuracy",accuracy_score(y_test,y_test_pred))





#Decision tree
model = DecisionTreeClassifier(
    criterion = "entropy",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
    )
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
print(f"\nAccuracy : {acc*100:.2f}%\n")

print("Confusion Matrx (rows = Actual, cols=Predicted):")
print(pd.DataFrame(cm,
                   index=["Actual_NoDisease", "Actual_Disease"],
                   columns=["Pred_NoDisease", "Pred_Disease"]))
plot_tree(
        model,
        feature_names=x.columns,
        class_names=["0","1"],
        filled=True,
        rounded=True,
        fontsize=8
)
plt.title("Decision Tree(Heart disease Dataset)")
plt.show()


#SVM
linear_svm = SVC(kernel = "linear")
linear_svm.fit(x_train,y_train)
linear_predictions = linear_svm.predict(x_test)
print("\nLinear Kernel Results:")
print("Accuracy:",accuracy_score(y_test,linear_predictions))
cm_linear = confusion_matrix(y_test,linear_predictions)
print("Confusion Matrix (Linear):\n", cm_linear)
rbf_svm = SVC(kernel="rbf")
rbf_svm.fit(x_train,y_train)

rbf_prediction = rbf_svm.predict(x_test)
print("\nRBF Kernel Result:")
print("Accuracy:", accuracy_score(y_test, rbf_prediction))
cm_rbf = confusion_matrix(y_test,rbf_prediction)



