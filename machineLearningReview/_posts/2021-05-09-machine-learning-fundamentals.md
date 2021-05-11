---
identifier: "01"
layout: post
title: "Machine Learning Fundamentals"
categories:
  - Machine Learning
tags:
  - Supervised Learning
  - Machine Learning
  - problems
  - datasets
last_modified_at: 2021-05-09
---

## Problems Machine Learning Can Solve

The most successful machine learning algorithms are those that automate decision-making processes by generelizing from known examples. In this setting, which is known as *Supervised Learning*, the user provides the algorithm with pairs of inputs and desired outputs, and the algorithm finds a way to produce the desired output. *Unsupervised Learning* are the other type of algorithm that only the input is known, and no known output data is given to the algorithm.

## Measuring Sucess: Training and Testing data

Before we can apply our model to new measurements, we need to know whether it actually works - that is, whether we should trust its predictions. First of all, we cannot use the data we used to build the model to evaluated it. This is because our model can always remember the whole training set, and will therefore always predict the correct label for any point in the training set. "Remembering" dont tell us whether our model **generelize** well. 

To test the model's performance, we show it new data for which we have labels. This is usually done by splitting the labeled data we collected into two parts. One part is used to build our machine learning model, called *training data* and the other part will be used to test the performance of our model, called *test data*.