# Big Data Analytics Project 3
Third and last project for Big Data Analytics course CIIC5995 @ UPRM.
The project consists predicting if tweets are:
- related to a disease 1
- unrelated to a disease 2
- ambiguous 0

utilizing keras and tensorflow.
These tweets were obtained in [Project2](https://github.com/ortizcapetta/BigDataP2) pver the course of 24 hours.
The tweets here contain the keywords flu,zika,diarrhea,ebola,headache and measles.

## Predicting Tweets
To start the program and obtain results,
```
python3 main.py
```
which will utilize the data located in [Data](/Data) to train ML models and predict the label of the provided tweets.
Our results can be found in the [results tab](/Results) which contains tweets and their predicted label (0,1 or 2).

## Spark analysis
Once the results are in csv files, we can process them in spark using:
```
python3 spark_analysis.py
```
This will simply print out the distribution of 0,1 and 2 in both models. These are then utilized in the visualizations.

## Visualization
Visualizations have been provided in the [Visualization tab](/Visualization). Provided are two HTML files
created in google charts featuring the distribution of 0,1 and 2 in both ML models utilized to predict labels.


