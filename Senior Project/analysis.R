
library(trelliscopejs)
library(tidyverse)
library(skimr)
library(caret)
set.seed(100)

library(readr)
library(mosaic)
library(pander)
library(car)
library(DT) 
library(resample)
local_dir <- getwd()
cleanfit <- read_rds("Data/cleanfit.rds") %>%
  select(-paleo_worked, -weight_watchers_worked, -view_on_exercise, -relationship)
cleanfit$weekly_intense_exercise <- as.integer(cleanfit$weekly_intense_exercise)
ggplot(cleanfit, aes(x=body_fat , fill =free_weights )) + geom_histogram(bin= 60) +
  xlim(0, 10) + ylim(0, 10) + theme_bw() +
  facet_trelliscope(~ time,  nrow = 1, ncol = 2, width = 500, path =local_dir)

str(cleanfit$free_weights)
model1 <- glm(free_weights~ ., data = cleanfit, family = "binomial")
summary(model1)

model2 <- lm(bmi~ ., data = cleanfit)
summary(model2)
levels(cleanfit$view_on_exercise)



rows1          <- sample(nrow(cleanfit))
cleanfit    <- cleanfit[rows1, ]
inTrain   <- createDataPartition(y = cleanfit$body_fat, p = .60, list = FALSE)
training1  <- cleanfit[ inTrain,]
testing1   <- cleanfit[-inTrain,]
#################################
mycontrol1 <- trainControl(
  method = "cv",
  number = 50)
model_foba <- train(
  body_fat ~.,
  data=training1,
  method = "foba",
  trControl = mycontrol1, 
  tuneLength= 50)
test_pred <- predict(model_foba, newdata = testing1)
confusionMatrix(test_pred, testing1$win )
varImp(model_foba)
p <- predict(model_foba, testing1)
colAUC(p, testing1[["body_fat"]], plotROC = TRUE)




