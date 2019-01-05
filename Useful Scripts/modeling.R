#### Libraries, seed, and data
devtools::install_github("BYUIDSS/BYUImachine")
library(caret)
library(ggplot2)
library(caretEnsemble)
library(ISLR)
library(tidyverse)
library(ada)
library(caTools)
library(gbm)
library(ipred)
library(e1071)
library(earth)
library(mda)
library(arm)
library(gam)
library(monmlp)
library(bst)
library(RSNNS)
library(HDclassif)
library(bnclassify)
library(ordinalNet)
library(stepPlr)
library(rotationForest)
library(rpartScore)
library(sda)
library(spls)
library(LiblineaR)
library(binda)
library(kerndwd)
library(h2o)
library(LogicReg)
library(nodeHarvest)
library(mxnet)
library(BYUImachine)

#### Data

set.seed(100)
method_as         <- read_csv("personal_folders/shaefferc/model_assignments.csv") %>%
  filter(assignment == "Spencer")
hs_pre                <- readRDS("raw_data/cogs_preprocessed.rds")


#### Randomize rows, then split the data into training and testing sets
rows              <- sample(nrow(hs_pre))
hs_pre            <- hs_pre[rows, ]
intrain           <- createDataPartition(y = hs_pre$Truth, p= 0.8, list = FALSE)
train             <- hs_pre[intrain,]
test              <- hs_pre[-intrain,]

#### Train Control for dirty data
#myControl <- trainControl(
  #method = "cv",
  #number = 10,
  #summaryFunction = twoClassSummary,
  #classProbs = TRUE, # IMPORTANT!
  #verboseIter = TRUE,
  #preProc = c("knnImpute", "center", "scale", "nzv"))

#### Train Control for clean data
myControl <- trainControl(
  method          = "cv",
  number          = 2,
  #search          = "random",
  #summaryFunction = twoClassSummary,
  classProbs      = TRUE,
  verboseIter     = TRUE)

#### Logistic Regression
model_logistic <- train(
  Truth ~.,
  data=train,
  method = "glm",
  trControl = myControl)
test_pred <- predict(model_logistic, newdata = test)
confusionMatrix(test_pred, test$Truth )
varImp(model_logistic)
write_rds(model_logistic, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_logistic.rds")
model_logistic$coefnames



#### Random Forest
model_randomforest <- train(
  Truth~.,
  tuneLength = 1,
  data = train,
  method = "rf",
  importance=TRUE,
  trControl = myControl
)
test_pred1 <- predict(model_randomforest, newdata = test)
confusionMatrix(test_pred1, test$Truth )
varImp(model_randomforest)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### Glmnet
model_glmnet <- train(
  Truth~., train,
  tuneGrid = expand.grid(alpha = 0:1,
  lambda = seq(0.0001, 1, length = 20)),
  method = "glmnet",
  trControl = myControl
)
test_pred2 <- predict(model_glmnet, newdata = test)
confusionMatrix(test_pred2, test$Truth )
varImp(model_glmnet)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Linear Discriminant Analysis
model_lda <- train(
  Truth~.,
  train,
  method="lda",
  trControl=myControl
)
test_pred3 <- predict(model_lda, newdata = test)
confusionMatrix(test_pred3, test$Truth )
varImp(model_lda)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Boosted Trees
model_boosted <- train(
  Truth~.,
  train,
  method="ada",
  trControl=myControl
)
test_pred4 <- predict(model_boosted, newdata = test)
confusionMatrix(test_pred4, test$Truth )
varImp(model_boosted)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Naive Bayes
model_naivebayes <- train(
  Truth~.,
  train,
  method="naive_bayes",
  trControl=myControl
)
test_pred5 <- predict(model_naivebayes, newdata = test)
confusionMatrix(test_pred5, test$Truth )
varImp(model_naivebayes)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### kNN

model_knn <- train(
  Truth~.,
  tuneLength = 1,
  data = train,
  method = "knn",
  trControl = myControl
)
test_pred6 <- predict(model_knn, newdata = test)
confusionMatrix(test_pred6, test$Truth )
varImp(model_knn)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Learning Vector Quantization
model_lvq <- train(
  Truth~.,
  tuneLength = 1,
  data = train,
  method = "lvq",
  trControl = myControl
)
test_pred7 <- predict(model_lvq, newdata = test)
confusionMatrix(test_pred7, test$Truth )
varImp(model_lvq)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Support Vector Machine
svm_Linear <- train(
  Truth ~.,
  data = train,
  method = "svmLinear",
  trControl=myControl,
  tuneLength = 1)
varImp(svm_Linear)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Bagged CART
treebag <- train(
  Truth ~.,
  data = train,
  method = "treebag",
  trControl = myControl,
  tuneLength = 1)
test_pred7 <- predict(treebag, newdata = test)
confusionMatrix(test_pred7, test$Truth )
varImp(treebag)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Bagged Flexible Discriminant Analysis
Bagged_DA <- train(
  Truth ~.,
  data = train,
  method = "bagFDA",
  trControl = myControl,
  tuneLength = 1)
test_pred7 <- predict(Bagged_DA, newdata = test)
confusionMatrix(test_pred7, test$Truth )
varImp(Bagged_DA)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Bagged MARS
Bagged_MARS <- train(
  Truth ~.,
  data = train,
  method = "bagEarth",
  trControl = myControl,
  tuneLength = 1)
test_pred7 <- predict(Bagged_MARS, newdata = test)
confusionMatrix(test_pred7, test$Truth )
varImp(Bagged_MARS)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Bayesian Generalized Linear Model
Bayes_glm <- train(
  Truth ~.,
  data = train,
  method = "bayesglm",
  trControl = myControl,
  tuneLength = 1)
test_pred7 <- predict(Bayes_glm, newdata = test)
confusionMatrix(test_pred7, test$Truth )
varImp(Bayes_glm)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Boosted Generalized Linear Model
glmboost <- train(
  Truth ~.,
  data = train,
  method = "glmboost",
  trControl = myControl,
  tuneLength = 1)
test_pred7 <- predict(glmboost, newdata = test)
confusionMatrix(test_pred7, test$Truth )
varImp(glmboost)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### C5.0
model_c5 <- train(
  Truth ~.,
  data = train,
  method = "C5.0",
  trControl = myControl,
  tuneLength = 1)
test_pred7 <- predict(model_c5, newdata = test)
confusionMatrix(test_pred7, test$Truth )
varImp(model_c5)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### CART
model_CART <- train(
  Truth ~.,
  data = train,
  method = "rpart1SE",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_CART, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_CART)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### CART2
model_CART2 <- train(
  Truth ~.,
  data = train,
  method = "rpart2",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_CART2, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_CART2)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### C Forest
model_cforest <- train(
  Truth ~.,
  data = train,
  method = "cforest",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_cforest, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_cforest)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Conditional Inference Tree
model_CITree <- train(
  Truth ~.,
  data = train,
  method = "ctree2",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_CITree, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_CITree)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Generalized Additive Model using Splines ---- Not working
model_gam <- train(
  Truth ~.,
  data = train,
  method = "gam",
  trControl = trainControl(method="LOOCV", number =1),
  tuneLength =1,
  tuneGrid = data.frame(method="GCV.Cp", select = FALSE))
test_pred8 <- predict(model_gam, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_gam)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### Generalized Additive Model using LOESS - NOT ACCURATE
model_gamloess <- train(
  Truth ~.,
  data = train,
  method = "gamLoess",
  trControl = myControl,
  tuneLength =3)
test_pred8 <- predict(model_gamloess, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_gamloess)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Gaussian Process with Radial Basis Function Kernel - Not working
model_gaussian <- train(
  Truth ~.,
  data = train,
  method = "gaussprRadial",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_gaussian, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_gaussian)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### Multivariate Adaptive Regression Splines
model_gcvearth <- train(
  Truth ~.,
  data = train,
  method = "gcvEarth",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_gcvearth, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_gcvearth)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### Generalized Linear Model with Stepwise Feature Selection
model_glmstepAIC <- train(
  Truth ~.,
  data = train,
  method = "glmStepAIC",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_glmstepAIC, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_glmstepAIC)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### Heteroscedastic Discriminant Analysis - Does not work
model_hda <- train(
  Truth ~.,
  data = train,
  method = "hda",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_hda, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_hda)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### High Dimensional Discriminant Analysis
model_hdda <- train(
  Truth ~.,
  data = train,
  method = "hdda",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_hdda, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_hdda)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### Linear Discriminant Analysis
model_lda2 <- train(
  Truth ~.,
  data = train,
  method = "lda2",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_lda2, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_lda2)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Mixture Discriminant Analysis
model_mda <- train(
  Truth ~.,
  data = train,
  method = "mda",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_mda, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_mda)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Multi-Layer Perceptron, multiple layers
model_mlpml <- train(
  Truth ~.,
  data = train,
  method = "mlpWeightDecayML",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_mlpml, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_mlpml)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")

#### Monotone Multi-Layer Perceptron Neural Network
model_mono <- train(
  Truth ~.,
  data = train,
  method = "monmlp",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_mono, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_mono)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### Monotone Multi-Layer Perceptron Neural Network
model_nb <- train(
  Truth ~.,
  data = train,
  method = "nb",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_nb, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_nb)
write_rds(model_randomforest, "C:/Users/nic-g/Desktop/Data Science/hs_machinelearning/models/model_randomforest.rds")


#### Semi-Naive Structure Learner Wrapper - Does not work
model_nbsearch <- train(
  Truth ~.,
  data = train,
  method = "nbSearch",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_nbsearch, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_nbsearch)


#### Penalized Ordinal Regression
model_ordinalNet <- train(
  Truth ~.,
  data = train,
  method = "ordinalNet",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_ordinalNet, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_ordinalNet)



#### Penalized Logistic Regression
model_plr <- train(
  Truth ~.,
  data = train,
  method = "plr",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_plr, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_plr)


#### Partial Least Squares
model_pls <- train(
  Truth ~.,
  data = train,
  method = "pls",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_pls, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_pls)


#### Random Forest
model_ranger <- train(
  Truth ~.,
  data = train,
  method = "ranger",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_ranger, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_ranger)


#### Rotation Forest
model_rotationforest <- train(
  Truth ~.,
  data = train,
  method = "rotationForest",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_rotationforest, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_rotationforest)


#### Rotation Forest
model_rotationforest <- train(
  Truth ~.,
  data = train,
  method = "rotationForest",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_rotationforest, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_rotationforest)


#### CART
model_CART <- train(
  Truth ~.,
  data = train,
  method = "rpart2",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_CART, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_CART)


#### CART or Ordinal Responses - Does not work
model_rpartScore <- train(
  Truth ~.,
  data = train,
  method = "rpartScore",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_rpartScore, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_rpartScore)


#### Shrinkage Discriminant Analysis
model_sda <- train(
  Truth ~.,
  data = train,
  method = "sda",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_sda, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_sda)


#### Shrinkage Discriminant Analysis
model_spls <- train(
  Truth ~.,
  data = train,
  method = "spls",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_spls, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_spls)


#### Support Vector Machines with Boundrange String Kernel - Not working
model_svmBoundrangeString <- train(
  Truth ~.,
  data = train,
  method = "svmBoundrangeString",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_svmBoundrangeString, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_svmBoundrangeString)


#### L2 Regularized Support Vector Machine (dual) with Linear Kernel - Does not work
model_svmLinear3 <- train(
  Truth ~.,
  data = train,
  method = "svmLinear3",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_svmLinear3, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_svmLinear3)



#### eXtreme Gradient Boosting - Does not work
model_xgbDART <- train(
  Truth ~.,
  data = train,
  method = "xgbDART",
  trControl = myControl,
  tuneLength =1)
test_pred8 <- predict(model_xgbDART, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_xgbDART)


#### Binary Discriminant Analysis - Does not work
model_binda <- train(
  Truth ~.,
  data = train,
  method = "binda",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_binda, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_binda)


#### Boosted Smoothing Spline
model_bstsm <- train(
  Truth ~.,
  data = train,
  method = "bstSm",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_bstsm, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_bstsm)


#### Linear Distance Weighted Discrimination
model_dwdlinear <- train(
  Truth ~.,
  data = train,
  method = "dwdLinear",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_dwdlinear, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_dwdlinear)

#### Multivariate Adaptive Regression Spline
model_earth <- train(
  Truth ~.,
  data = train,
  method = "earth",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_earth, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_earth)

#Not working = gbm_h2o, gamboost

#### Boosted Generalized Linear Model
model_glmboost<- train(
  Truth ~.,
  data = train,
  method = "glmboost",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_glmboost, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_glmboost)


#### Logic Regression - doesn't work
model_logreg<- train(
  Truth ~.,
  data = train,
  method = "logreg",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_logreg, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_logreg)


#### Multi-Layer Perceptron - not accurate
model_mlp<- train(
  Truth ~.,
  data = train,
  method = "mlp",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_mlp, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_mlp)


#### Neural Network - Doesn't work
model_mxnet<- train(
  Truth ~.,
  data = train,
  method = "mxnet",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_mxnet, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_mxnet)


#### Neural Network - Doesn't work
model_mxnetAdam<- train(
  Truth ~.,
  data = train,
  method = "mxnetAdam",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_mxnetAdam, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_mxnetAdam)



#### Neural Network - doesn't work
model_nnet<- train(
  Truth ~.,
  data = train,
  method = "nnet",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_nnet, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_nnet)


#### Tree-Based Ensembles
model_nodeHarvest<- train(
  Truth ~.,
  data = train,
  method = "nodeHarvest",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_nodeHarvest, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_nodeHarvest)


#### Part DSA
model_partDSA<- train(
  Truth ~.,
  data = train,
  method = "partDSA",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_partDSA, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_partDSA)


#### Tree-Based Ensembles
model_protoclass<- train(
  Truth ~.,
  data = train,
  method = "protoclass",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_protoclass, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_protoclass)







#### Learning Vector Quantization - fatal error
model_lvq<- train(
  Truth ~.,
  data = train,
  method = "lvq",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_lvq, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_lvq)


#### Multilayer Perceptron Network with Dropout - not working
model_mlpKerasDropoutCost<- train(
  Truth ~.,
  data = train,
  method = "mlpKerasDropoutCost",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_mlpKerasDropoutCost, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_mlpKerasDropoutCost)



#### Multi-Layer Perceptron, with multiple layers - 82 percent
model_mlpML<- train(
  Truth ~.,
  data = train,
  method = "mlpML",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_mlpML, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_mlpML)


#### Multi-Layer Perceptron
model_mlpWeightDecay<- train(
  Truth ~.,
  data = train,
  method = "mlpWeightDecay",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_mlpWeightDecay, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_mlpWeightDecay)


#### Naive Bayes Classifier - not working
model_nbDiscrete<- train(
  Truth ~.,
  data = train,
  method = "nbDiscrete",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_nbDiscrete, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_nbDiscrete)


#### Quadratic Discriminant Analysis - not working
model_qda<- train(
  Truth ~.,
  data = train,
  method = "qda",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_qda, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_qda)


#### Radial Basis Function Network - Fatal error
model_rbf<- train(
  Truth ~.,
  data = train,
  method = "rbf",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_rbf, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_rbf)


#### Random Forest Rule-Based Model
model_rfRules<- train(
  Truth ~.,
  data = train,
  method = "rfRules",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_rfRules, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_rfRules)


#### Regularized Linear Discriminant Analysis - not working
model_rlda<- train(
  Truth ~.,
  data = train,
  method = "rlda",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_rlda, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_rlda)


#### ROC-Based Classifier
model_rocc<- train(
  Truth ~.,
  data = train,
  method = "rocc",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_rocc, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_rocc)


#### Rotation Forest
model_rotationForestCp<- train(
  Truth ~.,
  data = train,
  method = "rotationForestCp",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_rotationForestCp, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_rotationForestCp)


#### Stabilized Nearest Neighbor Classifier - not working
model_snn<- train(
  Truth ~.,
  data = train,
  method = "snn",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_snn, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_snn)


#### Linear Discriminant Analysis with Stepwise Feature Selection - not working
model_stepLDA<- train(
  Truth ~.,
  data = train,
  method = "stepLDA",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_stepLDA, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_stepLDA)



#### Quadratic Discriminant Analysis with Stepwise Feature Selection
model_stepQDA<- train(
  Truth ~.,
  data = train,
  method = "stepQDA",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_stepQDA, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_stepQDA)



#### Support Vector Machines with Linear Kernel
model_svmLinear2<- train(
  Truth ~.,
  data = train,
  method = "svmLinear2",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_svmLinear2, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_svmLinear2)




#### Support Vector Machines with Polynomial Kernel
model_svmPoly<- train(
  Truth ~.,
  data = train,
  method = "svmPoly",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_svmPoly, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_svmPoly)



#### Support Vector Machines with Class Weights
model_svmRadialWeights<- train(
  Truth ~.,
  data = train,
  method = "svmRadialWeights",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_svmRadialWeights, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_svmRadialWeights)




#### Tree Augmented Naive Bayes Classifier - not working
model_tan<- train(
  Truth ~.,
  data = train,
  method = "tan",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_tan, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_tan)





#### Tree Augmented Naive Bayes Classifier Structure Learner Wrapper - not working
model_tanSearch<- train(
  Truth ~.,
  data = train,
  method = "tanSearch",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_tanSearch, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_tanSearch)





#### Variational Bayesian Multinomial Probit Regression - not working
model_vbmpRadial<- train(
  Truth ~.,
  data = train,
  method = "vbmpRadial",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_vbmpRadial, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_vbmpRadial)


#### Partial Least Squares - 82 percent
model_widekernelpls<- train(
  Truth ~.,
  data = train,
  method = "widekernelpls",
  trControl = myControl,
  tuneLength =1,
  preProcess = c("YeoJohnson"))
test_pred8 <- predict(model_widekernelpls, newdata = test)
confusionMatrix(test_pred8, test$Truth)
varImp(model_widekernelpls)






#snippet trn
${1} <- train(
  x = train_x,
  y = train_y,
  method    = "${1}",
  trControl = myControl)

test_pred <- predict(${1}, newdata = test)
confusionMatrix(test_pred, test\$Truth)
varImp(${1})
write_rds(${1}, "../hs_models/${1}.rds")
remove(${1}, test_pred)



























#############################################################################



#### Creating a list of models
model_list <- caretList(
  Truth~., data=train,
  trControl=myControl,
  methodList=c("glm", "glmnet")
)


#### Putting models into a list, comparing the ROC curves, stacking the models
resamples <- resamples(model_list)
summary(resamples)
bwplot(resamples, metric = "ROC")
stack <- caretStack(model_list, method = "glm")

#### Creating an ensemble of models
greedy_ensemble <- caretEnsemble(
  model_list, 
  metric="ROC",
  trControl=myControl)
summary(greedy_ensemble)

p <- as.data.frame(predict(model_list, newdata=head(test)))
print(p)


#### Optional list with hyperparameters tweaked
model_list_big <- caretList(
  Truth~., data=training,
  trControl=my_control,
  metric="ROC",
  methodList=c("glm", "rpart"),
  tuneList=list(
    rf1=caretModelSpec(method="rf", tuneGrid=data.frame(.mtry=2)),
    rf2=caretModelSpec(method="rf", tuneGrid=data.frame(.mtry=10), preProcess="pca"),
    nn=caretModelSpec(method="nnet", tuneLength=2, trace=FALSE)
  )
)
#### XY plot
xyplot(resamples(model_list))
#### Correlation between models
modelCor(resamples(model_list))

#### Checking new accuracy, doesn't improve accuracy much
model_preds <- lapply(model_list, predict, newdata=test, type="prob")
model_preds <- lapply(model_preds, function(x) x[,"Healthy"])
model_preds <- data.frame(model_preds)
ens_preds <- predict(greedy_ensemble, newdata=test, type="prob")
model_preds$ensemble <- ens_preds
caTools::colAUC(model_preds, test$Truth)
varImp(greedy_ensemble)

glm_ensemble <- caretStack(
  model_list,
  method="glm",
  metric="ROC",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)
model_preds2 <- model_preds
model_preds2$ensemble <- predict(glm_ensemble, newdata=test, type="prob")
CF <- coef(glm_ensemble$ens_model$finalModel)[-1]
colAUC(model_preds2, test$Truth)

CF/sum(CF)

#### Playing with xgBoostExplainer
library(data.table)
library(rpart)
library(rpart.plot)
library(caret)
library(xgboost)
library(pROC)

rows <- sample(nrow(cogs_preprocessed))
cogs_preprocessed <- cogs_preprocessed[rows, ]
split <- round(nrow(cogs_preprocessed)* .80)
train <- cogs_preprocessed[1:split, ]
test <- cogs_preprocessed[(split + 1):nrow(cogs_preprocessed), ]

cv <- createFolds(train[1:35], k = 10)

myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
  )


tree.cv <- train(Truth ~.,
                 data = train,
                 method = "rpart2",
                 tuneLength = 7,
                 trControl = myControl,
                 control = rpart.control()
                 )

tree.model = tree.cv$finalModel
rpart.plot(tree.model, type = 2,extra =  7,fallen.leaves = T)
rpart.plot(tree.model, type = 2,extra =  2,fallen.leaves = T)



tree.preds = predict(tree.model, test)[,2]
tree.roc_obj <- roc(test$Truth, tree.preds)


cat("Tree AUC", auc(tree.roc_obj))

train$Truth <- as.numeric(as.factor(train$Truth)) - 1


xgb.train.data = xgb.DMatrix(data.matrix(train[1:35]), label = train$Truth, missing = NA)



param <- list(objective = "binary:logistic", base_score = 0.5)

str(train$Truth)


xgboost.cv = xgb.cv(params=param, data = xgb.train.data, folds = cv, nrounds = 1500, early_stopping_rounds = 100)
         

best_iteration = xgboost.cv$best_iteration


xgb.model <- xgboost(param =param,  data = xgb.train.data, nrounds=best_iteration)

xgb.test.data = xgb.DMatrix(data.matrix(test[1:35]), missing = NA)
xgb.preds = predict(xgb.model, xgb.test.data)
xgb.roc_obj <- roc(test$Truth, xgb.preds)

cat("Tree AUC ", auc(tree.roc_obj))
cat("XGB AUC ", auc(xgb.roc_obj))

col_names = attr(xgb.train.data, ".Dimnames")[[2]]
imp = xgb.importance(col_names, xgb.model)
xgb.plot.importance(imp)

library(xgboostExplainer)
explainer = buildExplainer(xgb.model,xgb.train.data, type="binary", base_score = 0.5, trees_idx = NULL)
pred.breakdown = explainPredictions(xgb.model, explainer, xgb.test.data)

testt <- test[1:35]

cat('Breakdown Complete','\n')
weights = rowSums(pred.breakdown)
pred.xgb = 1/(1+exp(-weights))
cat(max(xgb.preds-pred.xgb),'\n')
idx_to_get = as.integer(802)
test[idx_to_get,(1:35)]
showWaterfall(xgb.model, explainer, xgb.test.data, data.matrix(test[1:35]) ,idx_to_get, type = "binary")

plot(test$Truth, pred.breakdown$Truth, cex=0.4, pch=16, xlab = "Satisfaction Level", ylab = "Satisfaction Level impact on log-odds")
plot(test$Truth, pred.breakdown$Truth, cex=0.4, pch=16, xlab = "Last evaluation", ylab = "Last evaluation impact on log-odds")
cr <- colorRamp(c("blue", "red"))
plot(test$Truth, pred.breakdown$Truth, cex=0.4, pch=16, xlab = "Last evaluation", ylab = "Last evaluation impact on log-odds")
