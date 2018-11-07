##### Function  -----------------------------------------------------------------------------------------------------
model <- function(x, 
                  data = readRDS("raw_data/cogs_preprocessed.rds")
) {
  
  #### Required packages ----------------------------------------------------------------------------------------------
  require(caret, quietly = T)
  require(caretEnsemble,    quietly = T)
  require(ISLR,     quietly = T)
  require(tidyverse,  quietly = T)
  require(caTools,    quietly = T)
  
  
  #### Setting up the data split --------------------------------------------------------------------------------------
  rows <- sample(nrow(data))
  data <- data[rows, ]
  message("Shuffling the rows!")
  intrain <- createDataPartition(y = data$Truth, p= 0.8, list = FALSE)
  message("Splitting the data!")
  train <- data[intrain,]
  message("Setting up the training set!")
  test <- data[-intrain,]
  message("Setting up the testing set!")
  
  #### Setting up the train control object ----------------------------------------------------------------------------
  myControl <- trainControl(
    method = "cv",
    number = 10,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    verboseIter = TRUE)
  message("Setting up the train control object!")
  
  #### Fitting the model ----------------------------------------------------------------------------------------------
  message("Fitting the model")
  model_x <- train(
    Truth ~.,
    data=train,
    method = x,
    trControl = myControl,
    tuneLength = 10)
  test_pred <- predict(model_x, newdata = test)
  message("Finished!")
  message("Here is the confusion matrix!")
  print(confusionMatrix(test_pred, test$Truth))
  message("Here are the most important variables!")
  print(varImp(model_x))
  write_rds(model_x, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/personal_folders/stovern/model.rds")
  x <- model_x
  
  return(x)
  
  
  
  
  
}
