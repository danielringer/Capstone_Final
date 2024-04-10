## Detecting Air Travel Delays

**Gary Ringer**

### Executive summary
The intent of this project is to show the best machine learning model that, when using a set number of dataset features, could predict when a flight is most likely to be delayed. The use case would be for both individuals and businesses to know when to avoid booking a flight, or if a flight is neccessary, to know if a budget increase may be neccessary.

#### Rationale
In 2023, over 1.3 million domestic flights in the United States were delayed. That is just over 20% of total scheduled flights. Flight delays cause stress on travels due to increased travel times and increased expenses on both food and lodging. Businesses are also greatly impacted due to business resources delayed for important meetings and absorbing the increased costs mentioned.   
  
#### Research Question
Using a flight dataset, find the most efficient features and determine what the best machine learning model is to predict flight delays within the United States.

#### Data Sources
The dataset originates from Kaggle [here](https://www.kaggle.com/datasets/monareyhanii/flights/data)

This is a collection of flight data containing over 300,000 records from 2013. The dataset contains several features that are relevant to help determine causes for a delay.

#### Methodology

**Dataset Analysis**
The dataset contains 21 different features made of integers, floats, and objects. There are some missing values most closely associated with arrival times, departure times, and time in the air. Due to the increased size of the dataset, we are removing the rows with missing values. Being we are attempting to predictive delays, our target variable will be arr_delay.

**Data Understanding and Preparation**
The id and year columns have no relevance to research question and is removed from the dataset.
The name and carrier column delivers the same information. We removed the name column.
We find that the sched_dep_time column contains both the hour and minute columns and so remove it. Addtionally, the time_hour column contains the date (which we have in the month and day column) and the time (which we have in the sched_dep_time). It is also removed.
The month and day columns are independently, relatively equal. These will likely not be features.

![Flights_month_hist](https://github.com/danielringer/Capstone_20/assets/61809794/cc017eb4-57c5-4348-b2e8-4ca484b2927a)

![Flights_day_hist](https://github.com/danielringer/Capstone_20/assets/61809794/1f00c09d-72c0-44a3-b691-86d63df400ee)

![heatmap](https://github.com/danielringer/Capstone_20/assets/61809794/096cf7f5-954a-44ec-b7dc-f8abb6839796)

![delay_scatter](https://github.com/danielringer/Capstone_20/assets/61809794/ee4af4ae-a82a-454a-b54f-98d3ffa757cc)

We find that the flight number will be of little value and remove it.
The remaining object type columns are carrier, origin, and destination. We encode these strings as numbers.
The dataset now contains 14 columns.

**Feature Selection**
Using the continuos target variable of arr_delay, we use Mutual Info Regression to determine the best feature set. This is done by determining the amount of information one feature gives about another feature.

dep_delay : : 0.5227661922512521  
arr_time : : 0.2713665576980766  
dep_time : : 0.08424643906888019  
arr_delay : : 0.05181259734575683  
sched_arr_time : : 0.04328327005650756  
flight : : 0.04258818817168386  
sched_dep_time : : 0.04077437337558987  
air_time : : 0.037129841756029514  
origin_logReg : : 0.032199430226321546  
distance : : 0.022856231347652134  
month : : 0.022529250630472397  
day : : 0.012191358068675662  
carrier_logReg : : 0.005138609972116903  

We will use the features with a score of 4% and higher. 

dep_delay
arr_time
dep_time
sched_arr_time
flight
sched_dep_time

Target variable: arr_delay

**Lasso and Ridge Regression**

![ridge_lasso_table](https://github.com/danielringer/Capstone_20/assets/61809794/aea5936e-5a48-40dc-ae3c-1321f0a19521)

![ridge_scatter](https://github.com/danielringer/Capstone_20/assets/61809794/7804bcf0-281b-4813-9ef8-c1f5632e8de5)

![ridge_lasso_bar](https://github.com/danielringer/Capstone_20/assets/61809794/bce396ac-a3c7-42d7-961c-441a0e3796f8)

The Ridge Regression model barely beat out the Lasso Regression model with an MSE of 324 and R-Squared of .84.

**Data Improvement**
Using information derived from FAA.gov, we took the target variable, arr_delay, and created four categories to include, on time, General Delay, Moderate Delay, and Severe Delay.
We expect this to improve future models.

![flight_status_bar](https://github.com/danielringer/Capstone_20/assets/61809794/dff6aa21-4144-4473-8f0e-0c955aa94daa)

**Logistic Regression**
The Logistic Regression model had an accuracy score of 71% with 69,812 True positives.

![logistic_confusionMatrix](https://github.com/danielringer/Capstone_20/assets/61809794/de4e8655-2810-4cf9-b913-bebc81d49a57)

**K Nearest Neighbor**
The K Nearest Neighbor model, using GridSearchCV hyperparameters, had an accuracy score of 88% with 86,001 True positives.

![knn_confusionMatrix](https://github.com/danielringer/Capstone_20/assets/61809794/db4efa91-c0bd-4566-94ef-bd87fe8867ae)

**Decision Tree**
The Decision Tree model, using GridSearchCV hyperparameters, had an accuracy score of 91% with 89,747 True positives.

![dectree_confusionMatrix](https://github.com/danielringer/Capstone_20/assets/61809794/90903689-cc7f-404a-aaed-b95393d80058)

#### Results

![true_positives](https://github.com/danielringer/Capstone_20/assets/61809794/e41b2e75-30a5-4308-848f-423424435ad5)
![score_comparison](https://github.com/danielringer/Capstone_20/assets/61809794/f463e87b-bf61-4ac1-a117-fd9dfb0d5855)

Our research has determined the best features with the best model to be used for predicting flight delays in the United States. The Decision Tree Model was found to the best choice in determining delays.

#### Next steps
The dataset used was for the year 2013. Being that this data is over a decade old, the next step would be to collect more recent data. Next, I would suggest building pipelines that can support re-training for continuosly updated data.

#### Outline of project

Dataset: [Flights](https://www.kaggle.com/datasets/monareyhanii/flights/data)   
Notebook: [Link](https://github.com/danielringer/Capstone_20/blob/main/Airline_Delay_DataAnalysis_Modeling.ipynb)  
Technical Document: [Link](https://github.com/danielringer/Capstone_20/blob/main/README.md)  
Final Report: [Link](https://github.com/danielringer/Capstone_Final/blob/main/README.md)
