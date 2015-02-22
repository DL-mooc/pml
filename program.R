dt_raw <- read.csv("data/pml-training.csv", row.names=1)
dt_test <- read.csv("data/pml-testing.csv", row.names=1)

table(colSums(is.na(dt_raw)), colSums(is.na(dt_test)))
indexingColumns <- names(dt_raw)[1:6]
dt_compact <- dt_raw[, 
                     (colSums(is.na(dt_test)) == 0) 
                     & !(names(dt_raw) %in% indexingColumns)]
table(sapply(dt_compact, class))

library(caret)
set.seed(34656)
inTrain <- createDataPartition(y = dt_compact$classe, p = 0.7, list = FALSE)
training   <- dt_compact[inTrain, ]        # training set
validation <- dt_compact[-inTrain, ]       # validaton set



library(randomForest)
rfModel <- randomForest(classe ~ ., data = training)

predcv <- predict(rfModel, validation)
rfConfusion <- confusionMatrix(validation$classe, predcv)
print(rfConfusion)
# The out of sample error equals 1 minus accuracy (0.9963)
predt <- predict(rfModel, dt_test)
print(predt)

