library(ggplot2)
library(lubridate)
library(randomForest)

set.seed(1)

train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

library(randomForest)

extractFeatures <- function(data) {
  features <- c("season",
                "holiday",
                "workingday",
                "weather",
                "temp",
                "atemp",
                "humidity",
                "windspeed",
                "hour")
  data$hour <- hour(ymd_hms(data$datetime))
  return(data[,features])
}

trainFea <- extractFeatures(train)
testFea  <- extractFeatures(test)

submission <- data.frame(datetime=test$datetime, count=NA)

# We only use past data to make predictions on the test set
# so we train a new model for each test set cutoff point
for (i_year in unique(year(ymd_hms(test$datetime)))) {
  for (i_month in unique(month(ymd_hms(test$datetime)))) {
    cat("Year: ", i_year, "\tMonth: ", i_month, "\n")
    testLocs   <- year(ymd_hms(test$datetime))==i_year & month(ymd_hms(test$datetime))==i_month
    testSubset <- test[testLocs,]
    trainLocs  <- ymd_hms(train$datetime) <= min(ymd_hms(testSubset$datetime))
    rf <- randomForest(extractFeatures(train[trainLocs,]), train[trainLocs,"count"], ntree=100)
    submission[testLocs, "count"] <- predict(rf, extractFeatures(testSubset))
  }
}

write.csv(submission, file = "1_random_forest_submission.csv", row.names=FALSE)
