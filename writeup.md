# Human Activity Recognition Analysis
Daria Lidina  
Sunday, February 22, 2015  

This analysis builds a model to predict predict activity quality from activity monitors. Data for this analysis was downloaded from a course page on Coursera but the original source is http://groupware.les.inf.puc-rio.br/har.

## Getting and preprocessing data

The data for this analysis is assumed to be downloaded beforehand into `data` subdirectory of a working directory where the script is run.


```r
dt_raw <- read.csv("data/pml-training.csv", row.names=1)
dt_test <- read.csv("data/pml-testing.csv", row.names=1)
```

### Dealing with missing values

```r
table(colSums(is.na(dt_raw)), colSums(is.na(dt_test)))
```

```
##        
##          0 20
##   0     59 33
##   19216  0 67
```

Both datasets have some `NA`s in their columns. For the test set there are 100 columns consisting only of NA values. Obviously, that columns can't be used for prediction. The table also shows that all the columns having `NA`s in the training dataset also have `NA`s in testing dataset, so after removing those 100 the training dataset will no longer contain any `NA`s.

The first 6 columns of the dataset are attributes of particular observation, not possible predictors, they should also be removed.


```r
indexingColumns <- names(dt_raw)[1:6]
dt_compact <- dt_raw[, 
                     (colSums(is.na(dt_test)) == 0) 
                     & !(names(dt_raw) %in% indexingColumns)]
table(sapply(dt_compact, class))
```

```
## 
##  factor integer numeric 
##       1      25      27
```

We are running no more transformations on data, so it will be possible to apply the model built using the `dt_compact` datasets directly to the `dt_test`. 

## Building a model

First all the data is partitioned in training and validation sets. Validation data is set aside to only check the final model.

```r
library(caret)
set.seed(34656)
inTrain <- createDataPartition(y = dt_compact$classe, p = 0.7, list = FALSE)
training   <- dt_compact[inTrain, ]        # training set
validation <- dt_compact[-inTrain, ]       # validaton set
```

The model built in this analysis is a random forest model

```r
library(randomForest)
rfModel <- randomForest(classe ~ ., data = training)
```

## Model evaluation

Checking the model against the validation dataset.

```r
predcv <- predict(rfModel, validation)
rfConfusion <- confusionMatrix(validation$classe, predcv)
print(rfConfusion)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    5 1131    3    0    0
##          C    0   12 1013    1    0
##          D    0    0   12  952    0
##          E    0    0    0    2 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9941          
##                  95% CI : (0.9917, 0.9959)
##     No Information Rate : 0.2853          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9925          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9895   0.9854   0.9969   1.0000
## Specificity            1.0000   0.9983   0.9973   0.9976   0.9996
## Pos Pred Value         1.0000   0.9930   0.9873   0.9876   0.9982
## Neg Pred Value         0.9988   0.9975   0.9969   0.9994   1.0000
## Prevalence             0.2853   0.1942   0.1747   0.1623   0.1835
## Detection Rate         0.2845   0.1922   0.1721   0.1618   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9985   0.9939   0.9914   0.9972   0.9998
```

* The model accuracy is 0.9941 with 95% confidence.
* The out of sample error is 0.0059
* The P Value is small (< 2.2e-16), indicating a statistically significant test
* For all classes sensitivity and specificity are high (>98%).

## Predicting activities for test cases


```r
predt <- predict(rfModel, dt_test)
print(predt)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


