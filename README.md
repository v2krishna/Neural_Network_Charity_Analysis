# Neural_Network_Charity_Analysis

##  Purpose: 

The main purpose of the use case is to perform AI alogrithms using deep-learning neural networks with the TensorFlow platform in Python, to analyze and classify the success of charitable donations. The following is process used for the analysis:
  * preprocessing the data for the neural network model.
  * compile, train and evaluate the model.
  * optimize the model.
 
## Environment:
  * Data Source: charity_data.csv
  * Software: Python 3.7.7, Anaconda Navigator 1.9.12, Conda 4.8.4, Jupyter Notebook 6.0, TensorFlow

## Results:

### Processsing Steps:
  * Columns EIN and NAME are not useful to include in the analysis as they are more of KYC informaiton.
  * Drop the above mention columns from the feature selection.
  * IS_SUCCESSFUL is binary column determines whether the charity is been used effectively or not. we use this as our target variable for deep learning neural network.
  * The following columns APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are the features.
  * Encoding of the categorical variables, spliting into training and testing datasets and standardization have been applied to the features.

### Train Model & Evaluate:
  * Deep-Learning NN model is made of 2 hidden layers with 80 and 30 neurons respectively.
  * Input data has 43 features and 25,724 samples.
  * Output layer is made of a unique neuron as it is a binary classification.
  * The activation function ReLU is used for the hidden layers. As our output is a binary classification, Sigmoid is used on the output layer.
  * The model accuracy is under 75%. This is not a satisfying performance to help predict the outcome of the charity donations.
  * To increase the performance of the model, we applied bucketing to the feature ASK_AMT and organized the different values by intervals.
  * Increased the number of neurons on one of the hidden layers, then we used a model with three hidden layers.
  * Also tried a different activation function (tanh) but none of these steps helped improve the model's performance.

## Summary: 
 
The Deep Learning NN model did not reach the target of 75% accuracy.Based on the model performance we cannot use this model for predicting the charity donation been used effectively. Since this is classification situation, we could use a supervised machine learning model such as the Random Forest Classifier to combine a multitude of decision trees to generate a classified output and evaluate its performance against our deep learning model.

