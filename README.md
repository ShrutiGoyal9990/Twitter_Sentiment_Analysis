# **TWITTER SENTIMENT ANALYSIS**
# 1. Introduction
Sentiment Analysis, or Emotion AI is a natural language processing (NLP) approach that involves determining the sentiment or emotional tone expressed in a piece of text. It is used to recognize polarity in text, and hence evaluate the author's attitude towards a particular thing, administration, person, or place and determine whether it is negative, positive, irrelevant, or neutral.

## 2. About the Dataset
A dataset is a collection of data that is stored for a specific purpose. In this segment, I collected our data from the “Twitter Sentiment Analysis” Dataset available on Kaggle. There are a total of 69,491 tweets in the training dataset in which the positive values are “19,457”, negative values are “20,847” and other values are “29,187”. The primary confusion classes include positive, negative, neutral, and irrelevant. The validation dataset includes a total of 998 unique values. 

DATASET LINK : https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

## 3. Algorithms Used
## 3.1 Logistic Regression
Logistic Regression is a foundational principle in statistics and machine learning that plays a crucial role in binary and multi-class classification tasks. Contrary to popular belief, logistic regression is a classification algorithm rather than a regression algorithm. It is used to forecast the likelihood that a specific input instance belongs to a specific class. The result of logistic regression is a probability value that lies between 0 and 1 using the logistic function, also known as the sigmoid function.

![WhatsApp Image 2025-06-22 at 13 08 19_587ccf2b](https://github.com/user-attachments/assets/f13f06ab-e2fb-4e36-b847-9a3afd55201c)

## 3.2 XGBoost
XGBoost (Extreme Gradient Boosting) is a potent and popular machine learning algorithm that falls under the category of gradient boosting methods. It builds a powerful ensemble model out of the predictions of several smaller models, typically decision trees. The goal of each succeeding model is to fix the mistakes caused by the prior models. XGBoost can be effectively applied to sentiment analysis tasks. While it is not the most popular algorithm for sentiment analysis especially when dedicated models such as LSTM exist, it can still provide competitive results with appropriate feature engineering and parameter tuning. 

![WhatsApp Image 2025-06-22 at 13 09 26_5ae0f3aa](https://github.com/user-attachments/assets/413f22a2-18b3-4818-a959-20b11c3ae866)


## 3.3 Random Forest Classifier
It builds an ensemble of decision trees, where each tree is trained on a different subset of the given dataset. These decision trees work together to make predictions, and based on the majority number of votes the algorithm produces the final result. This model can handle various categories of data and is resistant to overfitting. Though it can provide competitive results, it is important to note that a Random Forest Classifier might not always be an excellent choice due to its incapability to capture complex linguistic nuances in a comprehensive manner.

![WhatsApp Image 2025-06-22 at 13 10 12_2cafa2ca](https://github.com/user-attachments/assets/c8b43687-448d-4b6f-912f-42fc35104978)


## 3.4 Support Vector Machine
The Support Vector Machine (SVM) is a supervised learning technique that primarily focuses on locating the best hyperplane in an N-dimensional space capable of dividing the data points into various classes while maximizing the margin between them. The margin is the distance between the hyperplane and the support vectors. SVM can perform both linear and non-linear classification by using different kernel functions. It offers several advantages especially when one is dealing with high dimensional data and complex decision boundaries however it has certain limitations as well such as sensitivity to noise, hyperparameter sensitivity, etc.

![WhatsApp Image 2025-06-22 at 13 11 10_ae77abff](https://github.com/user-attachments/assets/15478310-be8a-4f69-9a50-cedc61fbaef3)


## 3.5 Multinomial Naive Bayes
Natural language processing (NLP) and text classification applications frequently employ the probabilistic classification algorithm known as Multinomial Naive Bayes. It is a modification of the Naive Bayes algorithm created to deal with datasets that have discrete feature counts and numerous classes. It works by using the Bayes theorem to classify data into different classes based on the probabilities of observing specific features. Its suitability for features with discrete counts, such as word frequencies in text documents makes it a sensible choice for sentiment analysis, however similar to SVM, this algorithm is also sensitive to noise in the data.

![WhatsApp Image 2025-06-22 at 13 11 48_392a1faf](https://github.com/user-attachments/assets/4be09923-6910-42ec-b335-bac02a2f7b11)

## 5. Conclusion
This project exploits five supervised machine learning algorithms, stated as, Logistic Regression, XGBoost, Random Forest, Support Vector Machine, Multinomial Naive Bayes to successfully classify four sentiments (‘positive’, ‘negative’, ‘neutral’, ‘irrelevant’) among the total of 69,491 unique tweets.

Logistic Regression appears to be one of the most suited algorithms for this purpose as it showed the highest accuracy that is about 90.99%. 



