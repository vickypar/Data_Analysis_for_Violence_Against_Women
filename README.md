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

## 3. Our Approach

### 3.1 Data Selection
- Removal of features with missing values in all of the records
- Removal of features with the same value in all of the records
- Removal of irrelevant features (data source)
- Use the feature "OBS_COMMENT"

### 3.2 Data Preprocessing
- Visualize the per day closing price of the stock.
- Visualize the data in our series through a probability distribution.


### 3.3 Models
- Random Forest
- Logistic Regression
- Gradient Boosting Algorithm
- AdaBoost
- Neural Networks
- XGBoost
- k-Nearest Neighbors

### 3.4 Evaluation
 

## 4. Inference

Comparison between all approaches.


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
