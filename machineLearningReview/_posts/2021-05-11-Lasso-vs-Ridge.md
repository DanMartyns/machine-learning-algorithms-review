---
layout: post
identifier: 07
title: "Lasso vs Ridge Regression"
categories:
  - Supervised Learning
subcategory:
  - Linear Models
tags:
  - Supervised Learning
  - Linear Models
  - Regression
last_modified_at: {{ page.last_modified_at }}
---

In practice, ridge regression is usually the first choice between these two models. However, if you have a large amount of features and expect only a few of them to be important, Lasso might be a better choice. Similarly, if you would like to have a model that is easy to interpret, Lasso will provide a model that is easier to under‐ stand, as it will select only a subset of the input features. scikit-learn also provides the ElasticNet class, which combines the penalties of Lasso and Ridge. In practice, this combination works best, though at the price of having two parameters to adjust: one for the L1 regularization, and one for the L2 regularization.

## L1 Regularization vs L2 Regularization

In order to create a less complex model when you have a large number of features in a dataset are used Regularization techniques to address overfitting and feature selection.

As was said before, the regression model that use L1 Regularization technique is Lasso Regression and model which uses L2 Regularization technique is Ridge Regression. The key difference between these two is the penalty term. 

**Ridge Regression** adds "square magnitude" of coefficient as penalty term to the loss function. The second part of the formula bellow represents the L2 Regularization.

$$
\sum^n_{i=1}(y_i - \sum^p_{j=1}x_{ij}\Theta_j)^2 + \lambda\sum^p_{j=1}\Theta^2_j
$$

If the $$\lambda$$ elemet is 0, then we will get back the ordinary least square, whereas a very large value it will add to much wieght and it will lead to underfit. Hence, it's important how *lambda* is chosen. This technique works very well to avoid overfit issues.

**Lasso Regression** adds *"absolute value of magnitude"* of coefficient as penalty term to the loss function. 

$$
\sum^n_{i=1}(y_i - \sum^p_{j=1}x_{ij}\Theta_j)^2 + \lambda\sum^p_{j=1}|\Theta_j|
$$

Again, lambda zero it will get back de ordinary least square whereas very large value will make coefficients zero hence it will underfit.

The key different between this techniques is that Lasso shrinks the less important feature's coefficient to zero thus, removing some feature altogether. So, this works well for **feature selection** in case we have a huge number of features.