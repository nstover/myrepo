cogs <- readr::read_rds("raw_data/cogsDataDFratio2.rds") %>%
  ungroup() %>%
  select(-ID) %>%
  na.omit(Truth)


################ Step 1: Function initialization ##################
model_list <- function(x,   
  seed        = 100,         # Set the seed
  y           = cogs$Truth,  # Column of prediction
  method      = "cv",        # Method of resampling ("boot", "cv")
  number      = 10,
  methodList  = c("glm", "glmnet"),
  preProcess  = c("knnImpute", "nzv", "center", "scale")) {
  
  
################ Step 2:  Required packages  #####################
  require(tidyverse,       quietly = T)
  require(caret,           quietly = T)
  require(caretEnsemble,   quietly = T)
  require(mlbench,         quietly = T)
  require(rpart,           quietly = T)
  require(pROC,            quietly = T)
  

  
############## Step 3: Train Control #####################
  set.seed(seed)
  rows <- sample(nrow(x))
  data <- x[rows, ]
  inTrain <- createDataPartition(y, p = .8, list = FALSE)
  training <- x[ inTrain,]
  testing <- x[-inTrain,]
  my_control <- trainControl(
    method=method,
    number=number,
    savePredictions="final",
    classProbs=TRUE,
    index=createResample(training$y, 25),
    summaryFunction=twoClassSummary,
    preProcOptions = preProcess)

############### Step 4: List of models ####################
  
  model_list <- caretList(
    y~., data=training,
    trControl=my_control,
    metric="ROC",
    methodList=methodList
  )
  }

  
print(model_list)
model_list(cogs) 
model_list(cogs_preprocessed)
