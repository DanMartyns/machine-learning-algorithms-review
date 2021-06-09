---
layout: post
identifier: 14
title: "Uncertainty Estimates from Classifiers"
categories:
  - Supervised Learning
tags:
  - Supervised Learning
  - Decision Function
  - Predict Probabilities
  - Classification
---

Another useful part of the scikit-learn interface that we haven't talked about yet is the ability of classifiers to provide uncertainty estimates of predictions. Often, you are not only interested in which class a classifier predicts for a certain test point, but also how certain it is that this is the right class. In practice, different kinds of mistakes lead to very different outcomes in real-world applications. Imagine a medical application testing for cancer. Making a false positive prediction might lead to a patient undergoing additional tests, while a false negative prediction might lead to a serious disease not being treated.

There are two different functions in *scikit-learn* that can be used to obtain uncertainty estimates from classifiers: *decision_function* and *predict_proba*.

Most (but not all) classifiers have at least one of them, and many classifiers have both. Let’s look at what these two functions do on a synthetic two-dimensional dataset, when building a *GradientBoostingClassifier* classifier, which has both a *decision_function* and a *predict_proba* method:

```python   
In:
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles

X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

# we rename the classes "blue" and "red" for illustration purposes
y_named = np.array(["blue", "red"])[y]

# we can call train_test_split with arbitrarily many arrays;
# all will be split in a consistent manner
X_train, X_test, y_train_named, y_test_named, y_train, y_test =
 train_test_split(X, y_named, y, random_state=0)

# build the gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)
``` 

## The decision function

In the binary classification case, the return value of decision_function is of shape (n_samples,), and it returns one floating-point number for each sample:

```python   
In:
print("X_test.shape: {}".format(X_test.shape))
print("Decision function shape: {}".format(
 gbrt.decision_function(X_test).shape))
```

```text
Out:
X_test.shape: (25, 2)
Decision function shape: (25,)
```

This value encodes how strongly the model believes a data point to belong to the "positive" class, in this case class 1. Positive values indicate a preference for the positive class, and negative values indicate a preference for the "negative" (other) class:

```python   
In:
# show the first few entries of decision_function
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6]))
```

```text
Out:
Decision function:
[ 4.136 -1.683 -3.951 -3.626 4.29 3.662]
```

We can recover the prediction by looking only at the sign of the decision function:

```python   
In:
print("Thresholded decision function:\n{}".format(
 gbrt.decision_function(X_test) > 0))
print("Predictions:\n{}".format(gbrt.predict(X_test)))
```

```text
Out:
Thresholded decision function:
[ True False False False True True False True True True False True
 True False True False False False True True True True True False
 False]
Predictions:
['red' 'blue' 'blue' 'blue' 'red' 'red' 'blue' 'red' 'red' 'red' 'blue'
 'red' 'red' 'blue' 'red' 'blue' 'blue' 'blue' 'red' 'red' 'red' 'red'
 'red' 'blue' 'blue']
```

## Predicting Probabilities

The output of *predict_proba* is a probability for each class, and is often more easily understood than the output of *decision_function*. The first entry in each row is the estimated probability of the first class, and the second entry is the estimated probability of the second class. Because it is a probability, the output of *predict_proba* is always between 0 and 1, and the sum of the entries for both classes is always 1:

```python   
In:
# show the first few entries of predict_proba
print("Predicted probabilities:\n{}".format(
 gbrt.predict_proba(X_test[:6])))
```

```text
Out:
Predicted probabilities:
[[ 0.016 0.984]
 [ 0.843 0.157]
 [ 0.981 0.019]
 [ 0.974 0.026]
 [ 0.014 0.986]
 [ 0.025 0.975]]
```

Because the probabilities for the two classes sum to 1, exactly one of the classes will be above 50% certainty. That class is the one that is predicted. Because the probabilities are floating-point numbers, it is unlikely that they will both be exactly 0.500. However, if that happens, the prediction is made at random.

You can see in the previous output that the classifier is relatively certain for most points. How well the uncertainty actually reflects uncertainty in the data depends on the model and the parameters. **A model that is more overfitted tends to make more certain predictions, even if they might be wrong.** **A model with less complexity usually has more uncertainty in its predictions.** A model is called *calibrated* if the reported uncertainty actually matches how correct it is - in a calibrated model, a prediction made with 70% certainty would be correct 70% of the time.

## Uncertainty in Multiclass Classification

So far, we’ve only talked about uncertainty estimates in binary classification. But the *decision_function* and *predict_proba* methods also work in the multiclass setting. Let’s apply them on the Iris dataset, which is a three-class classification dataset:

```python   
In:
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
 iris.data, iris.target, random_state=42)

gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)

print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
# plot the first few entries of the decision function
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))
```

```text
Out:
Decision function shape: (38, 3)
Decision function:
[[-0.529 1.466 -0.504]
 [ 1.512 -0.496 -0.503]
 [-0.524 -0.468 1.52 ]
 [-0.529 1.466 -0.504]
 [-0.531 1.282 0.215]
 [ 1.512 -0.496 -0.503]]
```

In the multiclass case, the *decision_function* has the shape (n_samples, n_classes) and each column provides a "certainty score" for each class, where a large score means that a class is more likely and a small score means the class is less likely. You can recover the predictions from these scores by finding the maximum entry for each data point.

The output of *predict_proba* has the same shape, (n_samples, n_classes). Again, the probabilities for the possible classes for each data point sum to 1:

```python   
In:
# show the first few entries of predict_proba
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6]))

# show that sums across rows are one
print("Sums: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))
```

```text
Out:
Predicted probabilities:
[[ 0.107 0.784 0.109]
 [ 0.789 0.106 0.105]
 [ 0.102 0.108 0.789]
 [ 0.107 0.784 0.109]
 [ 0.108 0.663 0.228]
 [ 0.789 0.106 0.105]]
Sums: [ 1. 1. 1. 1. 1. 1.]
```

To summarize, *predict_proba* and *decision_function* always have shape (n_samples, n_classes) - apart from *decision_function* in the special binary case. In the binary case, *decision_function* only has one column, corresponding to the "positive" class classes_[1]. This is mostly for historical reasons.