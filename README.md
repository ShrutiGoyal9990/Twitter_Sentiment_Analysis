# **TWITTER SENTIMENT ANALYSIS**
# 1. INTRODUCTION 
In an era where technology has taken the world by storm, the digital age has completely transformed the way we communicate, connect, and acquire information with the world around us. However, beneath the synthetic reality, there lies the need for sentiment analysis.

Sentiment Analysis, or Emotion AI is a natural language processing (NLP) approach that involves determining the sentiment or emotional tone expressed in a piece of text. It is used to recognize polarity in text, and hence evaluate the author's attitude towards a particular thing, administration, person, or place and determine whether it is negative, positive, irrelevant, or neutral.

The dataset used in this research has been taken from Kaggle and various models such as Logistic regression, XGBoost, Random Forest Classifier, Support Vector Machine, Multinomial Naive Bayes and ensemble method have been implemented. We did the comparison among the above models on the basis of their precision, recall, and F1 score. Logistic Regression gave the highest accuracy of about 92.6%.

## 2. About the Dataset
A dataset is a collection of data that is stored for a specific purpose. In this segment, I collected our data from the “Twitter Sentiment Analysis” Dataset available on Kaggle. There are a total of 69,491 tweets in the training dataset in which the positive values are “19,457”, negative values are “20,847” and other values are “29,187”. The primary confusion classes include positive, negative, neutral, and irrelevant. The validation dataset includes a total of 998 unique values. 

## 3. Algorithms Used
## 3.1 Logistic Regression
Logistic Regression is a foundational principle in statistics and machine learning that plays a crucial role in binary and multi-class classification tasks. Contrary to popular belief, logistic regression is a classification algorithm rather than a regression algorithm. It is used to forecast the likelihood that a specific input instance belongs to a specific class. The result of logistic regression is a probability value that lies between 0 and 1 using the logistic function, also known as the sigmoid function.




![lr_ac](https://github.com/ShrutiGoyal9990/Twitter_Sentiment_Analysis/assets/121054868/277d0f3c-26d2-42d8-9a4f-af917244d32d)

![lr_cr](https://github.com/ShrutiGoyal9990/Twitter_Sentiment_Analysis/assets/121054868/14ffa3cb-1496-4ea9-9c63-e2fff687ba0c)

## 3.2 XGBoost
XGBoost (Extreme Gradient Boosting) is a potent and popular machine learning algorithm that falls under the category of gradient boosting methods. It builds a powerful ensemble model out of the predictions of several smaller models, typically decision trees. The goal of each succeeding model is to fix the mistakes caused by the prior models. XGBoost can be effectively applied to sentiment analysis tasks. While it is not the most popular algorithm for sentiment analysis especially when dedicated models such as LSTM exist, it can still provide competitive results with appropriate feature engineering and parameter tuning. 
![xg_ac](https://github.com/ShrutiGoyal9990/Twitter_Sentiment_Analysis/assets/121054868/d00736db-73fe-4534-895b-92a33905b1b7)
![xg_cr](https://github.com/ShrutiGoyal9990/Twitter_Sentiment_Analysis/assets/121054868/1c2fdff9-4147-4196-8643-f01f852740dc)

## 3.3 Random Forest Classifier
It builds an ensemble of decision trees, where each tree is trained on a different subset of the given dataset. These decision trees work together to make predictions, and based on the majority number of votes the algorithm produces the final result. This model can handle various categories of data and is resistant to overfitting. Though it can provide competitive results, it is important to note that a Random Forest Classifier might not always be an excellent choice due to its incapability to capture complex linguistic nuances in a comprehensive manner.
![rf_ac](https://github.com/ShrutiGoyal9990/Twitter_Sentiment_Analysis/assets/121054868/11ab3c13-a85b-4ef1-be20-8b308f2397eb)
![rf_cr](https://github.com/ShrutiGoyal9990/Twitter_Sentiment_Analysis/assets/121054868/fdf996ca-6433-4807-b75b-676c6f2a580c)

## 3.4 Support Vector Machine
The Support Vector Machine (SVM) is a supervised learning technique that primarily focuses on locating the best hyperplane in an N-dimensional space capable of dividing the data points into various classes while maximizing the margin between them. The margin is the distance between the hyperplane and the support vectors. SVM can perform both 
linear and non-linear classification by using different kernel functions. It offers several advantages especially when one is dealing with high dimensional data and complex decision boundaries however it has certain limitations as well such as sensitivity to noise, hyperparameter sensitivity, etc.
![sv_ac](https://github.com/ShrutiGoyal9990/Twitter_Sentiment_Analysis/assets/121054868/9822d62f-1926-4e82-b2e4-fbabbf2950b4)
![sv_cr](https://github.com/ShrutiGoyal9990/Twitter_Sentiment_Analysis/assets/121054868/f3bfc893-b0ea-48ea-ad6f-edba0872c8f3)

## 3.5 Multinomial Naive Bayes
Natural language processing (NLP) and text classification applications frequently employ the probabilistic classification algorithm known as Multinomial Naive Bayes. It is a 
modification of the Naive Bayes algorithm created to deal with datasets that have discrete feature counts and numerous classes. It works by using the Bayes theorem to classify data into different classes based on the probabilities of observing specific features. Its suitability for features with discrete counts, such as word frequencies in text documents makes it a sensible choice for sentiment analysis, however similar to SVM, this algorithm is also sensitive to noise in the data.
![nb_ac](https://github.com/ShrutiGoyal9990/Twitter_Sentiment_Analysis/assets/121054868/46f0ef12-1f40-4b51-91bc-cdab05efb9ae)
![nb_cr](https://github.com/ShrutiGoyal9990/Twitter_Sentiment_Analysis/assets/121054868/6f6c5a21-9ce9-4f58-9632-7f89bbab6834)






