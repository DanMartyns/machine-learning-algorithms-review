---
layout: post
identifier: 17
title: "Types of Data Represented as Strings"
categories:
  - Working with Text Data
tags:
  - types of data
---

Before we dive into the processing steps that go into representing text data for machine learning, we want to briefly discuss different kinds of text data that you might encounter. Text is usually just a string in your dataset, but not all string features should be treated as text. A string feature can sometimes represent categorical variables. There is no way to know how to treat a string feature before looking at the data.

There are four kinds of string data you might see: 
- Categorical data 
- Free strings that can be semantically mapped to categories 
- Structured string data 
- Text data

Categorical data is data that comes from a fixed list. Say you collect data via a survey where you ask people their favorite color through a dropdown menu with many colors. This will result in a dataset with exactly the colors that could be chosen, which encode a categorical feature.

Now imagine instead of providing a dropdown menu, it's provided a text field for the users to provide their favorite colors. Many people might respond with a color like "black" or "blue". Others might make typographical errors, use different spellings like "gray" and "grey", or use more evocate and specific names like "midnight blue". Also, it's possible to have very strange entries, such as "my dentist's office orange". This kind of answer can belong to the second category free strings that can be semantically mapped to categories. 

The final category of string data is freeform text data that consists of phrases or sentences. Examples include tweets, chat logs, and hotel reviews, as well as the collected works of Shakespeare, the content of Wikipedia, or the Project Gutenberg collection of 50,000 ebooks. All of these collections contain information mostly as sentences composed of words. Let's assume that all our documents are in the same language, English. In the context of text analysis, the dataset is often called the corpus, and each data point, represented as a single text, is called a document. These terms come from the *information retrieval (IR)* and *natural language processing (NLP)* communities, which both deal mostly with text data.

