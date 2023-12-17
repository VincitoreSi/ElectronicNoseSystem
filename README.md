# ElectronicNoseSystem
This repository contains my work for my B.Tech Project that is designing Electronic Nose System for Gas Sensor Array data. Project contain developing full pipeline which contains these steps primarily 
<!-- list -->
1. Collecting the data in the lab from sensors
2. Uploading the data on thingsPeak using ResbherryPi in real time
3. Analyzing data features using PCA, LDA and t-SNE plots etc.
4. Applying various Machine Learning approaches as well as Deep Learning algorithms on the data to classify the gases present in the mixture from sensors value
5. Using Regression based approaches predicting the concentration for the present gases in mixture.
6. Developing the API for automation of the whole process and real time visualization as well as prediction from the sesors value.
<!-- listend -->

# Applying Different Scikit-learn and Keras Based Classifiers on Gas Sensor Dataset
In this project, I collected two open source datasets and applied different machine learning classification techniques using python based Scikit-learn and Keras libraries. Response of various metal oxide gas sensors were collected while they were exposed to different gas mixtures. The aim is to find out and classify the individual gas components in the mixtures. In order to do so, various supervised machine learning classifiers were applied and their performance were compared.
## Dataset Description
Two different datasets were used in this project- One is for binary classification and anothere is for multi-class classification purpose. Both of them were collected from UCI machine learning repository. For a generalized overview of the dataset and different attribute information, head over to the associated links below-

* [Gas sensor array under dynamic gas mixtures Data Set (Binary Classification)](http://archive.ics.uci.edu/ml/datasets/gas+sensor+array+under+dynamic+gas+mixtures)
* [Gas Sensor Array Drift Dataset Data Set (Multi-class Classification)](https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset)

For classification purpose, I pre-processed and made the datasets simpler. The modified datasets can be found here:
[Dataset](https://drive.google.com/drive/folders/1gJy8f3twHl9rANqUMcxG1YMxvbV-cijN?usp=sharing)

## Data Processing Workflow
On both datasets, PCA and t-SNE dimension reduction techniques were applied in order to plot and visualize the relationships between different attributes.The same workflow was followed and the same classifiers were applied on both of the datasets. I applied 10 classical classifiers( non-neural network based) and 1 keras based vanilla neural network classifier. 

### Classical Classifiers
1. K-Nearest Neighbor (KNN)
2. Support Vector Machine (SVM)
3. Gaussian Multinomial Naive Bayes (MultinomialNB)
4. Decision Tree
5. Random Forest
6. Extra Tree
7. Logistic Regression
8. KNN based Bagging
9. Logistic Regression
10. Majority Voting Ensemble Machine

### Neural Network Based Classifier
I also created a 4 layered neural network architecture with 2 hidden layers for classification. The same network was used for both the binary and multi-class classification purposes. 