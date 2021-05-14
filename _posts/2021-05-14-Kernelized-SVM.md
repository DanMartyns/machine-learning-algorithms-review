---
layout: post
identifier: 12
title: "Kernelized Support Vector Machines"
categories:
  - Supervised Learning
tags:
  - Supervised Learning
  - Regression
  - Classification
  - Strenghts
  - Weaknesses
---

Kernelized support vector machines (often just referred to as SVMs) are an extension that allows for more complex models that are not defined simply by hyperplanes in the input space. While there are support vector machines for classification and regression, we will restrict ourselves to the classification case, as implemented in SVC.

## Linear Models and NonLinear Features

Linear Models can be quite limiting in low-dimensional spaces, as lines and hyperplanes have limited flexibility. One way to make a linear model more flexible is by adding more features - for example, by adding interactions or polynomials of the input features.

<img src="{{ site.url }}/{{site.baseurl}}/assets/images/015.png" alt="pair_plot" width="70%" style="margin: auto; display: block; "/>
<br />

A linear model for classification can only separate points using a line, and will not be able to do a very good job on this dataset. Let's now expanding the dataset dimension adding a new feature, as *$$(feature_1)^2$$*.

<div style="text-align: center;">
  <figure class="image">
  <div  style="display: flex; flex-direction: row;">
    <img src="{{ site.url }}/{{site.baseurl}}/assets/images/016.png" alt="pair_plot" width="50%"/>
    <img src="{{ site.url }}/{{site.baseurl}}/assets/images/017.png" alt="pair_plot" width="50%"/>
  </div>
    <figcaption style="font-size: 16px">Decision boundary found by a linear SVM on the expanded threedimensional dataset</figcaption>
  </figure>
</div>
<br />

In the new representation of the data, it is now possible to separate the two classes using a linear model, through a plane. However, as a function of the original features, the linear SVM model is not actually linear anymore. It is not a line, but more of an ellipse.

<img src="{{ site.url }}/{{site.baseurl}}/assets/images/018.png" alt="pair_plot" width="70%" style="margin: auto; display: block; "/>
<br />

## The kernel trick

The lesson here is that adding nonlinear features to the representation of our data, it is possible make linear models much more powerful. However, often we don't know which features to add, and adding many features might make computation very expensive. Luckily, there is a clever mathematical trick that allows us to learn a classifier in a higher-dimensional space without actually computing the new, possible very large representation. This is known as *kernel trick*, and it works by directly computing the distance (more precisely, the scalar products) of the data points for the expanded feature representation, without ever actually computing the expansion.

There are two ways to map your data into a higher-dimensional space that are commonly used with SVMs: **the polynomial kernel**, which computes all possible polynomials up to a certain degree of the original features; and the **radial basis function (RBF) kernel**, also known as the Guassian kernel.

## Understanding SVMs (page 112 book)