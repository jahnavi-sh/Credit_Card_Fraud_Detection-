# Credit_Card_Fraud_Detection-

This document is written to provide aid in understanding the project.

Contents of the document - 
1. Understanding the problem statement 
2.About the dataset
3. Machine learning 
4. Types of machine learning models with examples 
5. Machine learning algorithm used for the model - Logistic regression 
6. NumPy library 
7. Pandas library 
8. Scikit-learn library 
9. Exploratory data analysis 
10.Fixing missing values in the dataset 
11.Unbalanced data 
12.Undersampling 
13.Train-test split 
14.Model evaluation - accuracy 

What is the problem statement for the project?

This project is aimed at recognizing fraudulent credit card transactions so that the customers are not charged for items that they did not purchase. 
Here, we will try to tackle many challenges associated with building a credit card fraud detection system. 
Enormous data is processed everyday and model build must be fast enough to respond to scam in time.
Imbalance data because most of the transactions are legitimate and only a small number are fraudulent which makes it hard for detecting fraudulent ones. 
Data availability and collection are a challenge because the data is mostly private. 
Misclassified is another challenge because not every fraudulent transaction is caught and reported. 

About the dataset - 

The dataset consists of the following columns or features - 
1. Time - time of transaction 
2. Since credit card data is sensitive information. Therefore, rest of the data is encoded (V1 - V28) using principal component analysis and numerical data has been      provided. 
3. Amount - amount of transaction 
4. Class 

We are provided with unbalanced data. There are 2,84,315 legitimate transactions and 492 fraudulent transactions. 

0 - legitimate transaction 
1 - fraudulent transaction

Machine learning - 
Machine learning enables the processing of sonar signals and target detection. Machine Learning is a subset of Artificial Intelligence. This involves the development of computer systems that are able to learn by using algorithms and statistical measures to study data and draw results from it. Machine learning is basically an integration of computer systems, statistical mathematics and data.

Machine Learning is further divided into three classes - Supervised learning, Unsupervised learning and Reinforcement Learning. 

Supervised learning is a machine learning method in which models are trained using labelled data. In supervised learning, models need to find the mapping function and find a relationship between the input and output. In this, the user has a somewhat idea of what the output should look like. It is of two types - regression (predicts results with continuous output. For example, given the picture of a person, we have to predict their age on the basis of the given picture) and classification (predict results in a discrete output. For example, given a patient with a tumor, we have to predict whether the tumor is malignant or benign.) 

Unsupervised learning is a method in which patterns are inferred from the unlabelled input data. It allows us to approach problems with little or no idea what the results should look like. We can derive structure from the data where we don’t necessarily know the effect of variables. We can derive the structure by clustering the data based on relationships among the variables in the data. With unsupervised learning there is no feedback on the prediction results. It is of two types - clustering (model groups input data into groups that are somehow similar or related by different variables. For example, clustering data of thousands of genes into groups) and non-clustering (models identifies individual inputs. It helps us find structure in a chaotic environment. For example, the cocktail party problem where we need to identify different speakers from a given audiotape.)

Reinforcement learning is a feedback-based machine learning technique. It is about taking suitable action to maximise reward in a particular situation. For example, a robotic dog learning the movement of his arms or teaching self-driving cars how to depict the best route for travelling. 

For this project, I will use Logistic regression model. 

Regression models describe the relationship between variables by fitting a line to the observed data. Linear regression models use a straight line and logistic and non-linear regression models use a curved line. Regression allows to estimate how a dependent variable changes as the independent variables change. 

Logistic regression (or sigmoid function or logit function) is a type of regression analysis and is commonly used algorithm for solving binary classification problems. It predicts a binary outcome based on a series of independent variables. The output is a predicted probability, binary value rather than numerical value. If the predicted value is a considerable negative value, it’s considered close to zero. If the predicted value if a significant positive value, it’s considered close to one. The dependent variable generally follows bernoulli distribution. Unlike linear regression model, that uses ordinary least square for parameter estimation, logistic regression uses maximum likelihood estimation, gradient descent and stochastic gradient descent. There can be infinite sets of regression coefficients. The maximum likelihood estimate is that set of regression coefficients for which the probability of getting data we have observed is maximum. To determine the values of parameters, log of likelihood function is taken, since it does not change the properties of the function. The log-likelihood is differentiated and using iterative techniques like newton method, values of parameters that maximise the log-likelihood are determined. A confusion matrix may be used to evaluate the accuracy of the logistic regression algorithm. 

NumPy  
It is a python library used for working with arrays. It has functions for working in the domain of linear algebra, fourier transform, and matrices. It is the fundamental package for scientific computing with python. NumPy stands for numerical python. 

NumPy is preferred because it is faster than traditional python lists. It has supporting functions that make working with ndarray very easy. Arrays are frequently used where speed and resources are very important. NumPy arrays are faster because it is stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently. This is locality of reference in computer science. 

Pandas - 
Pandas is made for working with relational or labelled data both easily and intuitively. It provides various data structures and operations for manipulating numerical data and time series. 

It has a lot of advantages like - 
1. Fast and efficient for manipulating and analyzing data
2. Data from different file objects can be loaded 
3. Easy handling of missing data in data preprocessing 
4. Size mutability 
5. Easy dataset merging and joining 
6. Flexible reshaping and pivoting of datasets 
7. Gives time-series functionality 

Pandas is built on top of NumPy library. That means that a lot of structures of NumPy are used or replicated in Pandas. The data produced by pandas are often used as input for plotting functions of Matplotlib, statistical analysis in SciPy, and machine learning algorithms in Scikit-learn. 

Scikit-Learn - 

It provides efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction. It has numerous machine learning, pre-processing, cross validation, and visualization algorithms. 

Exploratory data analysis - 

‘describe()’ method returns description of data in DataFrame. It tells us the following information for each column - 
Count - number of non-empty values
Mean - the average (mean) value  
Std - standard deviation
Min - minimum value
25% - the 25 percentile 
50% - the 50 percentile 
75% - the 75 percentile
Max - maximum value

Missing values - 
Missing values are common when working with real-world datasets. Missing data could result from a human factor, a problem in electrical sensors, missing files, improper management or other factors. Missing values can result in loss of significant information. Missing value can bias the results of model and reduce the accuracy of the model. There are various methods of handling missing data but unfortunately they still introduce some bias such as favoring one class over the other but these methods are useful. 

In Pandas, missing values are represented by NaN. It stands for Not a Number. 

Reasons for missing values - 
1. Past data may be corrupted due to improper maintenance
2. Observations are not recorded for certain fields due to faulty measuring equipments. There might by a failure in recording the values due to human error. 
3.  The user has not provided the values intentionally. 

Why we need to handle missing values - 
1. Many machine learning algorithms fail if the dataset contains missing values. 
2. Missing values may result in a biased machine learning model which will lead to incorrect results if the missing values are not handled properly. 
3. Missing data can lead to lack of precision. 

Types of missing data - 
Understanding the different types of missing data will provide insights into how to approach the missing values in the dataset. 
1. Missing Completely at Random (MCAR) 
There is no relationship between the missing data and any other values observed or unobserved within the given dataset. Missing values are completely independent of other data. There is no pattern. The probability of data being missing is the same for all the observations. 
The data may be missing due to human error, some system or equipment failure, loss of sample, or some unsatisfactory technicalities while recording the values.
It should not be assumed as it’s a rare case. The advantage of data with such missing values is that the statistical analysis remains unbiased.   
2. Missing at Random (MAR)
The reason for missing values can be explained by variables on which complete information is provided. There is relationship between the missing data and other values/data. In this case, most of the time, data is not missing for all the observations. It is missing only within sub-samples of the data and there is pattern in missing values. 
In this, the statistical analysis might result in bias. 
3. Not MIssing at Random (NMAR)
Missing values depend on unobserved data. If there is some pattern in missing data and other observed data can not explain it. If the missing data does not fall under the MCAR or MAR then it can be categorized as MNAR. 
It can happen due to the reluctance of people in providing the required information. 
In this case too, statistical analysis might result in bias. 

How to handle missing values - 

isnull().sum() - shows the total number of missing values in each columns 

We need to analyze each column very carefully to understand the reason behind missing values. There are two ways of handling values - 
1. Deleting missing values - this is a simple method. If the missing value belongs to MAR and MCAR then it can be deleted. But if the missing value belongs to MNAR then it should not be deleted. 
The disadvantage of this method is that we might end up deleting useful data. 
You can drop an entire column or an entire row. 
2. Imputing missing values - there are various methods of imputing missing values
3. Replacing with arbitrary value 
4. Replacing with mean - most common method. But in case of outliers, mean will not be appropriate
5. Replacing with mode - mode is most frequently occuring value. It is used in case of categorical features. 
6. Replacing with median - median is middlemost value. It is better to use median in case of outliers. 
7. Replacing with previous value - it is also called a forward fill. Mostly used in time series data. 
8. Replacing with next value - also called backward fill. 
9. Interpolation 

value_counts() function - 
It is used to get a series containing counts of unique values. 

Parameters - 
1. Normalize - If True then the object returned will contain the relative frequencies of the unique values.	
2. Sort - Sort by frequencies.	
3. Ascending - Sort in ascending order.
4. Bins - Rather than count values, group them into half-open bins, only works with numeric data.	

Unbalanced classes - 
In classification cases, when the data available on one or more classes are extremely low, then it is a unbalanced class. 

This can be a problem because - 
We don’t get optimized results for the class which is unbalanced in real time as the algorithm model does not get sufficient insight at the underlying class. 
It creates a problem in making validation to test data because it is difficult to have representation across classes in case number of observations for few classes is extremely less. 

Following are some of the ways of handling it - 
1. Undersampling - Here, we randomly delete the class which has sufficient observations so that the comparative ration of two classes is significant in our data. This approach is simple but it can introduce a bias in the data because there is a high possibility that the data we are deleting may contain important information about the predictive class. 
2. Oversampling - For the unbalanced class randomly increase the number of observations which are just copies of existing samples. This ideally gives a sufficient number of samples to work with. However, oversampling may lead to overfitting to the training data. 
3. Synthetic sampling - synthetically manufacture observations of unbalanced classes which are similar to the existing using nearest neighbour classification. The problem comes when the number of observations are of extremely rare class. 

Train-test split - 
The entire dataset is split into training dataset and testing dataset. Usually, 80-20 or 70-30 split is done. The train-test split is used to prevent the model from overfitting and to estimate the performance of prediction-based algorithms. We need to split the dataset to evaluate how well our machine learning model performs. The train set is used to fit the model, and statistics of training set are known. Test set is for predictions. 

This is done by using scikit-learn library and train_test_split() function. 
Parameters - 
1. *arrays: inputs such as lists, arrays, data frames, or matrices
2. test_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our test size. its default value is none.
3. train_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our train size. its default value is none.
4. random_state: this parameter is used to control the shuffling applied to the data before applying the split. it acts as a seed.
5. shuffle: This parameter is used to shuffle the data before splitting. Its default value is true.
6. stratify: This parameter is used to split the data in a stratified fashion.

Model evaluation - 

Model evaluation is done to test the performance of machine learning model. It is done to determine whether the model is a good fit for the input dataset or not. 

In this case, we use accuracy. Accuracy is a performance metrics that is used to test a binary classification model. Accuracy measures the proportion of correct prediction to total data points.

Accuracy = ( tp + tn) / ( tp + fp + tn + fn )

Tp - true positive. This refers to the total number of observations that belong to the positive class and have been predicted correctly. 
Tn - true negatives. It is total number of observations that belong to the negative class and have been predicted correctly 
Fp - false positives. It total number of observations that have been predicted to belong to positive class, but instead belong to the negative class. 
Fn - false negatives. It is total number of observations that have been predicted to be a part of negative class but instead belong to the positive class. 
