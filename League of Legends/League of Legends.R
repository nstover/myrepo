library(readr)
library(skimr)
library(tidyverse)
library(caret)
library(trelliscopejs)
library(caretEnsemble)
library(caTools)
library(MASS)
library(doParallel)
library(dplyr)
#######################################
skim(small_game1)
local_dir <- getwd()
set.seed(100)
game                    <- read_csv("Data/stats1.csv")
game$win[game$win == 1] <- "win"
game$win[game$win == 0] <- "loss"
game                    <- game %>%
  select(-id) %>%
  as.tibble() 
any(is.na(game$win))
game$win                <- as.factor(game$win)
game$item1              <- as.factor(game$item1)
game$item2              <- as.factor(game$item2)
game$item3              <- as.factor(game$item3)
game$item4              <- as.factor(game$item4)
remove_cols             <- nearZeroVar(game, names = TRUE, 
                           freqCut = 2, uniqueCut = 20)
all_cols                <- names(game)
game                    <- game[ , setdiff(all_cols, remove_cols)]
#####################################
smallgametrain <- createDataPartition(y = game$win, p = .995, list = FALSE)
small_game1    <- game[-smallgametrain,]
small_game1 <- small_game1[-c(2,3,4,5)]
###################################
rows1          <- sample(nrow(small_game1))
small_game1    <- small_game1[rows1, ]
#################################
inTrain1   <- createDataPartition(y = small_game1$win, p = .60, list = FALSE)
training1  <- small_game1[ inTrain1,]
testing1   <- small_game1[-inTrain1,]
#################################
mycontrol1 <- trainControl(
  method = "cv",
  number = 10,
  #summaryFunction = twoClassSummary,
  classProbs = TRUE)
#########################
mycontrol2 <- trainControl(
  method = "cv",
  number = 10,
  #summaryFunction = twoClassSummary,
  classProbs = FALSE)
##############################
model_glm <- train(
  win ~.,
  data=training1,
  method = "glm",
  trControl = mycontrol1, 
  tuneLength= 10,
  preProcess = c("center", "scale"))
test_pred <- predict(model_glm, newdata = testing1)
confusionMatrix(test_pred, testing1$win )
varImp(model_glm)
p <- predict(model_glm, testing1, type="prob")
colAUC(p, testing1[["win"]], plotROC = TRUE)



qplot(goldearned, kills, data = small_game1) +
  xlim(0, 30000) + ylim(0, 30) + theme_bw() +
  facet_trelliscope(~ win, nrow = 1, ncol = 2, width = 500, path =local_dir)

model <- glm(win~ ., data = small_game1)
summary(model)
###################### Data is balanced
sum(small_game1$win == "loss")

sum(small_game1$win == "win")


as.character(small_game1$win)





















#### Random Forest
model_randomforest <- train(
  win~.,
  tuneLength = 1,
  data = training1,
  method = "rf",
  importance=TRUE,
  trControl = mycontrol1
)
test_pred1 <- predict(model_randomforest, newdata = testing1)
confusionMatrix(test_pred1, testing1$win )
varImp(model_randomforest)
summary(model_randomforest)
#### Glmnet
model_glmnet <- train(
  win~., training1,
  tuneGrid = expand.grid(alpha = 0:1,
                         lambda = seq(0.0001, 1, length = 20)),
  method = "glmnet",
  trControl = mycontrol1
)
test_pred2 <- predict(model_glmnet, newdata = testing1)
confusionMatrix(test_pred2, testing1$win )
varImp(model_glmnet)

#### Linear Discriminant Analysis
model_lda <- train(
  win~.,
  training1,
  method="lda",
  trControl=mycontrol1
)
test_pred3 <- predict(model_lda, newdata = testing1)
confusionMatrix(test_pred3, testing1$win )
varImp(model_lda)

#### Boosted Trees
model_boosted <- train(
  win~.,
  training1,
  method="ada",
  trControl=mycontrol1
)
test_pred4 <- predict(model_boosted, newdata = testing1)
confusionMatrix(test_pred4, testing1$win )
varImp(model_boosted)





#### Support Vector Machine
svm_Linear <- train(
  win ~.,
  data = training1,
  method = "svmLinear",
  trControl=mycontrol2,
  tuneLength = 10)
test_pred7 <- predict(svm_Linear, newdata = testing1)
confusionMatrix(test_pred7, testing1$win )
varImp(svm_Linear)

#### Bagged CART
treebag <- train(
  win ~.,
  data = training1,
  method = "treebag",
  trControl = mycontrol1,
  tuneLength = 10)
test_pred7 <- predict(treebag, newdata = testing1)
confusionMatrix(test_pred7, testing1$win )
varImp(treebag)

#### Bagged Flexible Discriminant Analysis - 81 (highest)
Bagged_DA <- train(
  win ~.,
  data = training1,
  method = "bagFDA",
  trControl = mycontrol1,
  tuneLength = 10)
test_pred7 <- predict(Bagged_DA, newdata = testing1)
confusionMatrix(test_pred7, testing1$win )
varImp(Bagged_DA)


#### Bayesian Generalized Linear Model
Bayes_glm <- train(
  win ~.,
  data = training1,
  method = "bayesglm",
  trControl = mycontrol1,
  tuneLength = 10)
test_pred7 <- predict(Bayes_glm, newdata = testing1)
confusionMatrix(test_pred7, testing1$win )
varImp(Bayes_glm)

#### Boosted Generalized Linear Model
glmboost <- train(
  win ~.,
  data = training1,
  method = "glmboost",
  trControl = mycontrol1,
  tuneLength = 10)
test_pred7 <- predict(glmboost, newdata = testing1)
confusionMatrix(test_pred7, testing1$win )
varImp(glmboost)

#### C5.0
model_c5 <- train(
  win ~.,
  data = training1,
  method = "C5.0",
  trControl = mycontrol1,
  tuneLength = 10)
test_pred7 <- predict(model_c5, newdata = testing1)
confusionMatrix(test_pred7, testing1$win )
varImp(model_c5)



#### Generalized Additive Model using LOESS - 81%
model_gamloess <- train(
  win ~.,
  data = training1,
  method = "gamLoess",
  trControl = mycontrol1,
  tuneLength =10)
test_pred8 <- predict(model_gamloess, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_gamloess)



#### Multivariate Adaptive Regression Splines - 82 %
model_gcvearth <- train(
  win ~.,
  data = training1,
  method = "gcvEarth",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_gcvearth, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_gcvearth)


#### Generalized Linear Model with Stepwise Feature Selection
model_glmstepAIC <- train(
  win ~.,
  data = training1,
  method = "glmStepAIC",
  trControl = mycontrol1,
  tuneLength =10)
test_pred8 <- predict(model_glmstepAIC, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_glmstepAIC)





#### Linear Discriminant Analysis
model_lda2 <- train(
  win ~.,
  data = training1,
  method = "lda2",
  trControl = mycontrol1,
  tuneLength =10)
test_pred8 <- predict(model_lda2, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_lda2)

#### Mixture Discriminant Analysis
model_mda <- train(
  win ~.,
  data = training1,
  method = "mda",
  trControl = mycontrol1,
  tuneLength =10)
test_pred8 <- predict(model_mda, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_mda)





#### Semi-Naive Structure Learner Wrapper - Does not work
model_nbsearch <- train(
  win ~.,
  data = training1,
  method = "nbSearch",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_nbsearch, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_nbsearch)


#### Penalized Ordinal Regression
model_ordinalNet <- train(
  win ~.,
  data = training1,
  method = "ordinalNet",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_ordinalNet, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_ordinalNet)



#### Penalized Logistic Regression
model_plr <- train(
  win ~.,
  data = training1,
  method = "plr",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_plr, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_plr)


#### Partial Least Squares
model_pls <- train(
  win ~.,
  data = training1,
  method = "pls",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_pls, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_pls)


#### Random Forest
model_ranger <- train(
  win ~.,
  data = training1,
  method = "ranger",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_ranger, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_ranger)


#### Rotation Forest
model_rotationforest <- train(
  win ~.,
  data = training1,
  method = "rotationForest",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_rotationforest, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_rotationforest)


#### Rotation Forest
model_rotationforest <- train(
  win ~.,
  data = training1,
  method = "rotationForest",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_rotationforest, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_rotationforest)


#### CART
model_CART <- train(
  win ~.,
  data = training1,
  method = "rpart2",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_CART, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_CART)


#### CART or Ordinal Responses - Does not work
model_rpartScore <- train(
  win ~.,
  data = training1,
  method = "rpartScore",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_rpartScore, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_rpartScore)


#### Shrinkage Discriminant Analysis
model_sda <- train(
  win ~.,
  data = training1,
  method = "sda",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_sda, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_sda)


#### Shrinkage Discriminant Analysis
model_spls <- train(
  win ~.,
  data = training1,
  method = "spls",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_spls, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_spls)


#### Support Vector Machines with Boundrange String Kernel - Not working
model_svmBoundrangeString <- train(
  win ~.,
  data = training1,
  method = "svmBoundrangeString",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_svmBoundrangeString, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_svmBoundrangeString)


#### L2 Regularized Support Vector Machine (dual) with Linear Kernel - Does not work
model_svmLinear3 <- train(
  win ~.,
  data = training1,
  method = "svmLinear3",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_svmLinear3, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_svmLinear3)



#### eXtreme Gradient Boosting - Does not work
model_xgbDART <- train(
  win ~.,
  data = training1,
  method = "xgbDART",
  trControl = mycontrol1,
  tuneLength =1)
test_pred8 <- predict(model_xgbDART, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_xgbDART)


#### Binary Discriminant Analysis - Does not work
model_binda <- train(
  win ~.,
  data = training1,
  method = "binda",
  trControl = mycontrol1,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_binda, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_binda)


#### Boosted Smoothing Spline
model_bstsm <- train(
  win ~.,
  data = training1,
  method = "bstSm",
  trControl = mycontrol1,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_bstsm, newdata = testing1)
confusionMatrix(test_pred8, testing1$win)
varImp(model_bstsm)
