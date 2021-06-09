---
layout: post
identifier: 16
title: "Building Pipelines"
categories:
  - Algorithm Chains and Pipelines
tags:
  - Algorithm Chains and Pipelines
---

Let’s look at how we can use the Pipeline class to express the workflow for training an SVM after scaling the data with MinMaxScaler (for now without the grid search). First, we build a pipeline object by providing it with a list of steps. Each step is a tuple containing a name (any string of your choosing ) and an instance of an estimator:

```python
In:
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])

pipe.fit(X_train, y_train)
```

Here, we created two steps: the first, called "scaler", is an instance of MinMaxScaler, and the second, called "svm", is an instance of SVC. Now, we can fit the pipeline, like any other scikit-learn estimator. Here, pipe.fit first calls fit on the first step (the scaler), then transforms the train‐ ing data using the scaler, and finally fits the SVM with the scaled data. To evaluate on the test data, we simply call pipe.score:

```python
In:
print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))
```
```text
Out:
Test score: 0.95
```

Calling the score method on the pipeline first transforms the test data using the scaler, and then calls the score method on the SVM using the scaled test data. Using the pipeline, we reduced the code needed for our “preprocessing + classification” process. The main benefit of using the pipeline, however, is that we can now use this single estimator in *cross_val_score* or *GridSearchCV*.

## Using Pipelines in Grid Searches