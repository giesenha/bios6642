---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Data loading/preparation

Recommend you use [Pandas](https://pandas.pydata.org/)

Other options:
* [NumPy](https://www.numpy.org/)

Scikit learn includes a number of publicly available datasets that can be used for learning ML. From the documentation:

***A dataset is a dictionary-like object that holds all the data and some metadata about the data. This data is stored in the `.data` member, which is a `n_samples, n_features` array. In the case of supervised problem, one or more response variables are stored in the `.target` member. More details on the different datasets can be found in the dedicated section.***

Some of the steps involved:
* Removing erroneous data
* Correcting errors
* Extracting parts of a corpus of data with automated tools.
* Integrating data from various sources
* Feature engineering/data enrichment
* Semantic mapping

**NOTE:** Most machine learning models/functions in Scikit expect data to be normalized (mean centered and scaled by the standard deviation times n_samples). Tree based methods do not usually require this.

These steps are often repeated multiple times as a project progresses - data visualization and modeling often result in more data preparation.

Data Cleaning takes 50 - 90% of a data scientists time:
* https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fgilpress%2Ffiles%2F2016%2F03%2FTime-1200x511.jpg
* https://dataconomy.com/2016/03/why-your-datascientist-isnt-being-more-inventive/

For more instruction, seee this excellent tutorial showing some examples of data loading, preparation, and cleaning: https://pythonprogramming.net/machine-learning-tutorial-python-introduction/

```python
# import  some of the libraries that we'll need
from sklearn import datasets
import numpy as np
import pandas as pd
```

Documentation for the Diabetes dataset is available at: https://scikit-learn.org/stable/datasets/index.html#diabetes-dataset

Columns in the dataset:
* Age
* Sex
* Body mass index
* Average blood pressure
* S1
* S2
* S3
* S4
* S5
* S6

**Each of these 10 feature variables have been mean centered and scaled by the standard deviation times n_samples (i.e. the sum of squares of each column totals 1).**

Target:
* A quantitative measure of disease progression one year after baseline

### Load dataset

```python
diabetes = datasets.load_diabetes()

with np.printoptions(linewidth=130):
    print('Data - first 5\n', diabetes.data[0:5,:])
    print('Target - first 5\n', diabetes.target[0:5])
```

```python
diabetes.target.shape
```

```python
diabetes.data.shape
```

```python
df = pd.DataFrame(data=diabetes.data, columns=['age', 'sex', 'bmi', 'abp', 's1', 's2', 's3', 's4', 's5', 's6'])
df['target'] = diabetes.target
df.head()
```

### Load human readable version of dataset

```python
# compare original data set to see what data looks like in native format
url="https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"
df=pd.read_csv(url, sep='\t')
# change column names to lowercase for easier reference
df.columns = [x.lower() for x in df.columns]
df.head()
```

```python
df.describe()
```

# Data visualization/exploration

Recommend you start with [Seaborn](http://seaborn.pydata.org/) - Makes matplotlib easier; can access any part of matplotlib if necessary. Other recommendations include:

* [matplotlib](https://matplotlib.org/) One of the older and more widespread in use
* [Altair](https://altair-viz.github.io/)
* [Bokeh](https://bokeh.pydata.org/en/latest/)
* [Plot.ly](https://plot.ly/python/)

```python
import seaborn as sns
sns.set()
sns.set_style("ticks", {
    'axes.grid': True,
    'grid.color': '.9',
    'grid.linestyle': u'-',
    'figure.facecolor': 'white', # axes
})
sns.set_context("notebook")
```

```python
sns.scatterplot(x=df.age, y=df.y, hue=df.sex, palette='Set1')
```

```python
sns.scatterplot(x=df.age, y=df.bmi, hue=df.sex, palette='Set1')
```

```python
sns.jointplot(x=df.age, y=df.bmi, kind='hex')
```

```python
tdf = df[df.sex == 1]
sns.jointplot(x=tdf.age, y=tdf.bmi, kind='hex')
```

```python
tdf = df[df.sex == 2]
sns.jointplot(x=tdf.age, y=tdf.bmi, kind='hex')
```

```python
sns.distplot(df.y, rug=True)
```

```python
sns.pairplot(df, hue="sex", palette='Set1')
```

### Load the matplotlib extension for interactivity

This will affect all subsequent plots, regardless of cell location.

Best to run this before any plotting in notebook

```python
# %matplotlib widget
```

```python
sns.scatterplot(x=df.age, y=df.bmi, hue=df.sex, palette='Set1')
```

# Machine learning

Recommend you use [scikit-learn](https://scikit-learn.org/stable/)

Deep Learning options:

* [Caffe](http://caffe.berkeleyvision.org/)
* [Fastai](https://docs.fast.ai/) - Simplifies deep learning similar to scikit-learn; based on PyTorch
* [Keras](https://keras.io/)
* [PyTorch](https://pytorch.org/)
* [TensorFlow](https://www.tensorflow.org/overview/)

Natural Language Processing options:

* [nltk](http://www.nltk.org/)
* [spaCy](https://spacy.io/)
* [Stanford NLP Libraries](https://nlp.stanford.edu/software/)

Computer Vision:
* [OpenCV](https://opencv.org/)

Forecasting/Time Series:

* [Prophet](https://facebook.github.io/prophet/)
* [statsmodels](https://www.statsmodels.org/stable/index.html) - Also does other statistical techniques and machine learning


## Regression

### Linear Regression

```python
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create linear regression object
regr = LinearRegression()

# by convention, X is features and y is target
# random_state: Set a number here to allow for same results each time
X_train, X_test, y_train, y_test = model_selection.train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

# Train the model using the training sets
regr.fit(X_train, y_train)
```

To see documentation on `train_test_split()`

```python
??model_selection.train_test_split
```

```python
# Make predictions using the testing set
y_pred = regr.predict(X_test)
```

```python
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
```

```python
# from https://stackoverflow.com/questions/26319259/sci-kit-and-regression-summary
import sklearn.metrics as metrics
def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    
regression_results(y_test, y_pred)
```

An `explained_variance` of `0.455` means that approximately 45% of the variance in the Target variable is explained by the linear regression formula

### Support Vector Machine Regression

The objective of this algorithm is to maximize the distance between the decision boundary and the samples that are closest to the decision boundary. Decision boundary is called the “Maximum Margin Hyperplane.” Samples that are closest to the decision boundary are the support vectors. Through mapping of the various dimensions of data (n) into higher dimensional space via a kernel function e.g. k(x,y) each individual maybe separated from its neighbor to better identify those classified into each category.


```python
# Create Support Vector Machine regression object
svm_regr = svm.SVR(gamma='auto')

# Train the model using the training sets
svm_regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = svm_regr.predict(X_test)

regression_results(y_test, y_pred)
```

### XGBoost Regression

XGBoost (eXtreme Gradient Boosting) is an algorithm that a few years ago was considered state of the art for applied machine learning and Kaggle competitions when dealing with structured data.

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

```python
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st

one_to_left = st.beta(10, 1)
from_zero_positive = st.expon(0, 50)

params = {  
    'n_estimators': st.randint(3, 40),
    'max_depth': st.randint(3, 40),
    'learning_rate': st.uniform(0.05, 0.4),
    'colsample_bytree': one_to_left,
    'subsample': one_to_left,
    'gamma': st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    'min_child_weight': from_zero_positive,
    'objective': ['reg:squarederror']
}

xgbreg = XGBRegressor(nthreads=-1)  
```

```python
gs = RandomizedSearchCV(xgbreg, params, n_jobs=1, cv=5, iid=False)  
gs.fit(X_train, y_train)
gs_pred = gs.predict(X_test)
gs
```

```python
regression_results(y_test, gs_pred)
```

## Classification

As we want to demonstrate classification (Target values are part of a class, not continuous numbers) we will switch to a different dataset. See https://scikit-learn.org/stable/datasets/index.html#breast-cancer-wisconsin-diagnostic-dataset for details.

Attribute Information:
 	
* radius (mean of distances from center to points on the perimeter)
* texture (standard deviation of gray-scale values)
* perimeter
* area
* smoothness (local variation in radius lengths)
* compactness (perimeter^2 / area - 1.0)
* concavity (severity of concave portions of the contour)
* concave points (number of concave portions of the contour)
* symmetry
* fractal dimension (“coastline approximation” - 1)

Class/Target:
* WDBC-Malignant
* WDBC-Benign


### Support Vector Machine Classification

```python
bc = datasets.load_breast_cancer()

with np.printoptions(linewidth=160):
    print('Data - first 5\n', bc.data[0:5,:])
    print('Target - first 5\n', bc.target[0:5])
```

```python
# by convention, X is features and y is target
# random_state: Set a number here to allow for same results each time
X_train, X_test, y_train, y_test = model_selection.train_test_split(bc.data, bc.target, test_size=0.2, random_state=42)

# Create Support Vector Machine Classifier object
svmc = svm.SVC(kernel='linear', gamma='auto')

# Train the model using the training sets
svmc.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = svmc.predict(X_test)

svmc
```

```python
print("Classification report for classifier %s:\n%s\n"
      % (svmc, metrics.classification_report(y_test, y_pred)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
```

```python
data = {'y_pred': y_pred,
        'y_test':    y_test
        }

df = pd.DataFrame(data, columns=['y_test','y_pred'])
confusion_matrix = pd.crosstab(df['y_test'], df['y_pred'], rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)
```

### XGBoost Classifier

```python
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score

xclas = XGBClassifier()
xclas.fit(X_train, y_train) 
xg_y_pred = xclas.predict(X_test)

cross_val_score(xclas, X_train, y_train)
```

```python
print("Classification report for classifier %s:\n%s\n"
      % (xclas, metrics.classification_report(y_test, xg_y_pred)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, xg_y_pred))
```

## Clustering (unlabeled data)

Principle Component Analysis (PCA) is a technique used to emphasize variation and bring out strong patterns in a dataset. It's often used to make data easy to explore and visualize as you can use it to find those variables that are most unique and just keep 2 or 3 which can then be easily visualized.

```python
from sklearn.decomposition import IncrementalPCA

X = bc.data
y = bc.target

n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)
```

```python
# if plot data in 2 dimensions, are there any obvious clusters?
sns.scatterplot(x=X_ipca[:, 0], y=X_ipca[:, 1], palette='Set1')
```

```python
# what if we label data by Target variable?
sns.scatterplot(x=X_ipca[y == 0, 0], y=X_ipca[y == 0, 1], palette='Set1')
sns.scatterplot(x=X_ipca[y == 1, 0], y=X_ipca[y == 1, 1], palette='Set1')
```

### K-Means clustering

This technique requires you to know the number of clusters when you start. Since you may not know the number of clusters, you can visually determine the number based on distortion. See https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# calculate distortion for a range of number of cluster
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
```

```python
from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=2,
    init='random',
    n_init=10,
    max_iter=300, 
    tol=1e-04,
    random_state=0
)
y_km = km.fit_predict(bc.data)
```

```python
# plot the 3 clusters
plt.scatter(
    bc.data[y_km == 0, 0], bc.data[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    bc.data[y_km == 1, 0], bc.data[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
```

## Understanding/Explaining the model

See:

* LIME (Local Interpretable Model-agnostic Explanations)
  * Github: https://github.com/marcotcr/lime 
  * Paper: https://arxiv.org/abs/1602.04938
* SHAP (SHapley Additive exPlanations)
  * Github: https://github.com/slundberg/shap
  * Paper: http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions


## Bonus: Deep Learning with structured data

Using Fastai Library and the Diabetes data set used for regression examples.

https://www.kaggle.com/magiclantern/deep-learning-structured-data
