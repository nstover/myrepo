#### Libraries, seed, and data
library(caret)
library(caretEnsemble)
library(ISLR)
library(tidyverse)
library(ada)
library(plyr)
library(caTools)
library(gbm)
set.seed(100)
cogs_preprocessed <- readRDS("raw_data/cogs_preprocessed.rds")
cogs <- readRDS("raw_data/cogsDataDFratio2.rds") %>%
  ungroup() %>%
  select(-ID) %>%
  filter(!is.na(Truth)) %>%
  na_if(Inf) %>%
  as.data.frame()


#### Randomize rows, then split the data into training and testing sets
rows <- sample(nrow(cogs_preprocessed))
cogs_preprocessed <- cogs_preprocessed[rows, ]
intrain <- createDataPartition(y = cogs_preprocessed$Truth, p= 0.8, list = FALSE)
train <- cogs_preprocessed[intrain,]
test <- cogs_preprocessed[-intrain,]

#### Train Control for dirty data
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  preProcess("knnImpute", "center", "scale", "nzv"))

#### Train Control for clean data
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE)

#### Logistic Regression
model_logistic <- train(
  Truth ~.,
  data=train,
  method = "glm",
  trControl = myControl)
test_pred <- predict(model_logistic, newdata = test)
confusionMatrix(test_pred, test$Truth )

#### Random Forest
model_randomforest <- train(
  Truth~.,
  tuneLength = 1,
  data = train, method = "ranger",
  trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE)
)
test_pred1 <- predict(model_randomforest, newdata = test)
confusionMatrix(test_pred1, test$Truth )

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

#### Linear Discriminant Analysis
model_lda <- train(
  Truth~.,
  train,
  method="lda",
  trControl=myControl
)
test_pred3 <- predict(model_lda, newdata = test)
confusionMatrix(test_pred3, test$Truth )

#### Boosted Trees
model_boosted <- train(
  Truth~.,
  train,
  method="ada",
  trControl=myControl
)
test_pred4 <- predict(model_boosted, newdata = test)
confusionMatrix(test_pred4, test$Truth )

#### Naive Bayes
model_naivebayes <- train(
  Truth~.,
  train,
  method="naive_bayes",
  trControl=myControl
)
test_pred5 <- predict(model_naivebayes, newdata = test)
confusionMatrix(test_pred5, test$Truth )

#### kNN

model_knn <- train(
  Truth~.,
  tuneLength = 20,
  data = train,
  method = "knn",
  trControl = myControl
)
test_pred6 <- predict(model_knn, newdata = test)
confusionMatrix(test_pred6, test$Truth )

#### Learning Vector Quantization
model_lvq <- train(
  Truth~.,
  tuneLength = 5,
  data = train,
  method = "lvq",
  trControl = myControl
)
test_pred7 <- predict(model_lvq, newdata = test)
confusionMatrix(test_pred7, test$Truth )

#### Support Vector Machine
svm_Linear <- train(
  Truth ~.,
  data = train,
  method = "svmLinear",
  trControl=myControl,
  tuneLength = 10)


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
