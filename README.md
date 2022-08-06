# Modelling-experimental-and-observational-data-in-Python
In this report, we have analysed the relationship between World Development Indicator and Covid-19 death rate. We have done clustering using K-means clustering algorithm by which we make 6 clusters, equals to number of  continents. We use logistic regression for binary classification to foresee high COVID causalities and get 100% accuracy. For multi class classification we use two more algorithms i.e. Linear Discriminant Analysis and  Quadratic discriminant analysis along with Logistic Regression and then compare their results, LDA gives  highest accuracy of 89.3% among all three of them.

Appendix:
 Import all necessary Libraries:
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
import warnings
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
%matplotlib inline
sns.set()
 Loading Dataset:
data = pd.read_csv('project.csv')
data['Covid.deaths']=pd.to_numeric(data['Covid.deaths'].str.replace(',',''))
data['Comp.education']=pd.to_numeric(data['Comp.education'].str.replace('..','0'))
data=data.replace(np.nan,0)
Project_data = data.copy()
Project_data.head()
 Checking Correlation:
cor = Project_data.corr()
plt.figure(figsize=(15,8))
sns.heatmap(cor,annot=True)

 Describing Dataset:
Project_data.describe().T
 I find an Insight and then I draw it:
plt.figure(figsize=(12,8))
plt.title('Life expactancy in different Continents according to the water resourser')
sns.scatterplot(x='Water.services',y='Life.expec',hue='Continent',data=Project_data,size='Pop.total',legend=False,s
izes=(20, 4000))
plt.show()
 Creating Histograms:
sns.set_context('poster')
plt.figure(figsize=(35,20))
for i in range(3,19):
 plt.subplot(3,6,i)
 plt.title(Project_data.columns[i])
 plt.hist(Project_data.iloc[:,i])
 plt.show()
 Creating boxplot:
plt.title("Box plot continent againast covid death rate\n")
boxplot = Project_data.boxplot(figsize = (4,19), rot = 90, fontsize= '5', grid = False)
plt.figure(figsize=(2,1))
sns.boxplot(y='Continent', x='Covid.deaths', data=Project_data, orient='h')
plt.show()
 Data preprocessing:
from sklearn.preprocessing import MinMaxScaler
print(Project_data)
X = Project_data.iloc[:,3:]
df = X.dropna()
scaler = MinMaxScaler()
new=scaler.fit_transform(df)
scaled=pd.DataFrame(columns=df.columns,data=new)

 Clustering:
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(scaled)
pred = scaled.copy()
pred['kmean1'] = kmeans.labels_
pred.head()
 For Optimal Numbers of Clusters:
sil = []
kmax = 6
 Dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be

for k in range(2, kmax+1):
 kmeans = KMeans(n_clusters = k).fit(df)
 labels = kmeans.labels_ 
 sil.append(silhouette_score(df, labels, metric = 'euclidean'))
plt.figure(figsize=(15,6))
plt.plot(sil)
sns.set_context('poster')
plt.xlabel('clusters')
plt.ylabel('score')
plt.title('The silhouette score')
plt.show()
Project_data = Project_data.dropna()
 Binary Classification With Logistic Regression
X, y = make_blobs(n_samples=1000, centers=2, random_state=1)
 summarize observations by class label
count = Counter(y)
 plot the dataset and color the by class label
for i in range(10):

for label, _ in count.items():
 row_ix = where(y == label)[0]
 plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.legend()
plt.show()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
 Create an instance of the model.
logregbinary = LogisticRegression()
 Training the model.
logregbinary.fit(X_train,y_train)
 Do prediction.
y_pred=logregbinary.predict(X_test)
 Analyzing the results
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
 name of classes
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
 create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

 Applying Logistic for multi classes:
from sklearn.metrics import classification_report, confusion_matrix
feature_cols =['Covid.deaths','Life.expec','Elect.access','Net.nat.income','Net.nat.income.capita',
 'Mortality.rate','Primary','Pop.growth','Pop.density','Pop.total','Health.exp.capita','Health.exp',
 'Unemployment','GDP.growth','GDP.capita','Birth.rate','Water.services','Comp.education']
X = Project_data[feature_cols].replace(",", "", regex=True) # Features
y = Project_data.iloc[:,2].replace(",", "", regex=True)
yy = pd.DataFrame(y)
y=pd.cut(yy['Covid.deaths'], bins=[0, 800, 1600, 2400,8000], include_lowest=True, labels=['low', 'mid', 'high','ve
ry high'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)
 instantiate the model (using the default parameters)
logreg = LogisticRegression()
 fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
from sklearn import metrics
cm = confusion_matrix(y_test, y_pred, labels=logreg.classes_)
plt.figure(figsize=(15,6))
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
 labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Logistic Regression Confusion matrix');
ax.xaxis.set_ticklabels(y.unique()); ax.yaxis.set_ticklabels(y.unique());
plt.show()
print("\n\nLogistic Regression")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred , average='micro'))
print("Recall:",metrics.recall_score(y_test, y_pred ,average='micro'))

 Applying QDA:
qda=QuadraticDiscriminantAnalysis(reg_param=0.95)
qda.fit(X_train,y_train)
pred_qda=qda.predict(X_test)
cm = confusion_matrix(y_test, pred_qda, labels=logreg.classes_)
plt.figure(figsize=(15,6))
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
 labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Quadratic Discriminant Analysis Confusion matrix');
ax.xaxis.set_ticklabels(y.unique()); ax.yaxis.set_ticklabels(y.unique());
plt.show()
print("\n\nQuadratic Discriminant Analysis")
print("Accuracy:",metrics.accuracy_score(y_test, pred_qda))
print("Precision:",metrics.precision_score(y_test, pred_qda , average='micro'))
print("Recall:",metrics.recall_score(y_test, pred_qda ,average='micro'))
 Applying LDA:
lda = LinearDiscriminantAnalysis()
lda = lda.fit(X_train, y_train)
pred_lda=lda.predict(X_test)
cm = confusion_matrix(y_test, pred_lda, labels=logreg.classes_)
plt.figure(figsize=(15,6))
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
 labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Linear Discriminant Analysis Confusion matrix');
ax.xaxis.set_ticklabels(y.unique()); ax.yaxis.set_ticklabels(y.unique());
plt.show()
print("\n\nLinear Discriminant Analysis")
print("Accuracy:",metrics.accuracy_score(y_test, pred_lda))

print("Precision:",metrics.precision_score(y_test, pred_lda , average='micro'))
print("Recall:",metrics.recall_score(y_test, pred_lda ,average='micro'))
