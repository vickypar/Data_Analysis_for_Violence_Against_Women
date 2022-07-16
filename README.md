# Data_Analysis_for_Violence_Against_Women
A dataset that contains records about violence against women in the Pacific ocean was studied and preprocessed and machine learning algorithms were trained and compared in Python in order to predict the impact of the violence on victim's life.

## Table of Contents

[0. Installation](https://github.com/vickypar/Data_Analysis_for_Violence_Against_Women#0-installation)

[1. About](https://github.com/vickypar/Data_Analysis_for_Violence_Against_Women#1-about)

[2. Data](https://github.com/vickypar/Data_Analysis_for_Violence_Against_Women#2-data)

[3. Our Approach](https://github.com/vickypar/Data_Analysis_for_Violence_Against_Women#3-our-approach)

[4. Inference](https://github.com/vickypar/Data_Analysis_for_Violence_Against_Women#4-inference)

[5. Licensing and References](https://github.com/vickypar/Data_Analysis_for_Violence_Against_Womenn#6-licensing-and-references)


## 0. Installation 

The code requires Python versions of 3.* and general libraries available through the Anaconda package.

## 1. About

**Data Analysis for Violence Against Women** is a project that was created as a semester Project in the context of “Machine Learning” class.
MSc Data and Web Science, School of Informatics, Aristotle University of Thessaloniki.


## 2. Data

The dataset that was used in this project comes from the data source "Pacific Data Hub". More specifically, the dataset "Violence Against Women - VAW" contains metrics about violence against women from records collected from 22 countries in the Pacific Ocean during the time period 2006-2019.

![image](https://user-images.githubusercontent.com/95586847/179351013-02f04033-b5a7-459a-8918-bbdd12772796.png)

The outcome of the violence against women in the labeled dataset is presented below.

![image](https://user-images.githubusercontent.com/95586847/179354455-fe31af33-e03b-409b-aacb-c26861525990.png)

## 3. Our Approach

### 3.1 Data Selection
- Removal of features with missing values in all of the records
- Removal of features with the same value in all of the records
- Removal of irrelevant features (data source)
- Use the feature "OBS_COMMENT" to fill in some missing values of other features

### 3.2 Data Preprocessing
- Fill in missing values using "Single Imputer"
- Create one-hot vectors using "OneHotEncoder"
- Scale numeric features using "StandardScaler"
- Apply Principal Component Analysis (PCA) to reduce dimensions

### 3.3 Models
Seven different classifiers were trained and tested using a validation set to select the most appropriate values for the hyperparameters.

- Random Forest

  ![image](https://user-images.githubusercontent.com/95586847/179354749-ea6db5e8-a8dc-4300-8dbd-cd6949246fc2.png)
  - 6 estimators were selected
- Logistic Regression
  - 'lbfgs' solver
  - 1000 iterations
- Gradient Boosting Algorithm

  ![image](https://user-images.githubusercontent.com/95586847/179354760-d53f92a7-784b-4155-ad5a-1e73c952ff3c.png)
  - 40 estimators were selected

- AdaBoost

  ![image](https://user-images.githubusercontent.com/95586847/179354801-1939e4ed-f2a6-41bf-b110-9e1f751c1fc6.png)
  - 25 estimators were selected
  - 4 was the maximum depth of each tree
- Neural Networks
  - 2 hidden layers
  - 50 nodes
  - 'lbfgs' solver
  - 1000 iterations
- XGBoost

  ![image](https://user-images.githubusercontent.com/95586847/179354895-f4af5849-053d-4154-adb3-90a84d3a18aa.png)
  - 3 was the maximum depth of each tree
  - 40 estimators were selected
- k-Nearest Neighbors

  ![image](https://user-images.githubusercontent.com/95586847/179354942-f49bdc03-63a3-4b5c-a17a-d943fa1863e4.png)
  - k = 50

### 3.4 Evaluation
Comparison between all approaches in terms of Accuracy and Precision. 

![image](https://user-images.githubusercontent.com/95586847/179354716-1dffe114-58cc-45d6-bd76-4ae4e6dc8a7e.png)

## 4. Inference
Adaboost classifier was selected to predict the outcome in the unlabeled dataset. The results are presented below.

![image](https://user-images.githubusercontent.com/95586847/179354976-1206aa20-6b4d-41a7-89d0-d3036276caa9.png)

![image](https://user-images.githubusercontent.com/95586847/179354981-223ea7ac-13ec-4d5c-a268-017d0b49bf72.png)

## 5. Licensing and Refferences

Copyright (c) 2022 vickypar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
