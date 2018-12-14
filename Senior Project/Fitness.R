library(readr)
library(tidyverse)
library(skimr)
library(caret)
set.seed(100)
local_dir <- getwd()
fitdata                                       <- read_csv("Data/finalfitnessdata.csv")

fitdata$sex                                   <- as.factor(fitdata$sex)
fitdata$relationship                          <- as.factor(fitdata$relationship)
fitdata$health_rating                         <- as.factor(fitdata$health_rating)
fitdata$experience                            <- as.factor(fitdata$experience)
fitdata$weekly_intense_exercise               <- as.factor(fitdata$weekly_intense_exercise)
fitdata$workout_frequency                     <- as.factor(fitdata$workout_frequency)
fitdata$walking                               <- as.factor(fitdata$walking)
fitdata$running                               <- as.factor(fitdata$running)
fitdata$yoga                                  <- as.factor(fitdata$yoga)
fitdata$free_weights                          <- as.factor(fitdata$free_weights)
fitdata$strength_circuit                      <- as.factor(fitdata$strength_circuit)
fitdata$resistance_training                   <- as.factor(fitdata$resistance_training)
fitdata$biking                                <- as.factor(fitdata$biking)
fitdata$swimming                              <- as.factor(fitdata$swimming)
fitdata$conditioning                          <- as.factor(fitdata$conditioning)
fitdata$sports                                <- as.factor(fitdata$sports)
fitdata$feeling_better                        <- as.factor(fitdata$feeling_better)
fitdata$seeing_results                        <- as.factor(fitdata$seeing_results)
fitdata$having_fun                            <- as.factor(fitdata$having_fun)
fitdata$`praise/rewards`                      <- as.factor(fitdata$`praise/rewards`)
fitdata$accountability                        <- as.factor(fitdata$accountability)


fitdata$obstacles                             <- as.factor(fitdata$obstacles)
fitdata$goals                                 <- as.factor(fitdata$goals)
fitdata$sports_frequency                      <- as.factor(fitdata$sports_frequency)
fitdata$exercise_plan                         <- as.factor(fitdata$exercise_plan)
fitdata$workout_more                          <- as.factor(fitdata$workout_more)
fitdata$pre_workout                           <- as.factor(fitdata$pre_workout)
fitdata$protein                               <- as.factor(fitdata$protein)
fitdata$vitamins                              <- as.factor(fitdata$vitamins)

fitdata$weight_watchers_tried                 <- as.factor(fitdata$weight_watchers_tried)
fitdata$keto_tried                            <- as.factor(fitdata$keto_tried)
fitdata$intermittent_fasting_tried            <- as.factor(fitdata$intermittent_fasting_tried)
fitdata$iifym_tried                           <- as.factor(fitdata$iifym_tried)
fitdata$none_tried                            <- as.factor(fitdata$none_tried)
fitdata$paleo_tried                           <- as.factor(fitdata$paleo_tried)
fitdata$none_worked                           <- as.factor(fitdata$none_worked)
fitdata$weight_watchers_worked                <- as.factor(fitdata$weight_watchers_worked)
fitdata$intermittent_fasting_worked           <- as.factor(fitdata$intermittent_fasting_worked)
fitdata$iifym_worked                          <- as.factor(fitdata$iifym_worked)
fitdata$keto_worked                           <- as.factor(fitdata$keto_worked)
fitdata$paleo_worked                          <- as.factor(fitdata$paleo_worked)

fitdata$morning_hunger                        <- as.factor(fitdata$morning_hunger)
fitdata$meal_services                         <- as.factor(fitdata$meal_services)
fitdata$meal_plan_freedom                     <- as.factor(fitdata$meal_plan_freedom)
fitdata$view_on_exercise                      <- as.factor(fitdata$view_on_exercise)
fitdata$time                                  <- as.factor(fitdata$time)


fitdata1 <- fitdata %>%
  ungroup() %>%
  select(-Timestamp, -paleo_worked, -weight_watchers_worked) %>%
  mutate(bmi = (weight*0.453592)/((height * 0.0254)^2)) %>%
  as.tibble()


cleanfit <- fitdata1
 

write_rds(cleanfit, "Data/cleanfit.rds")
cleanfit <- separate(fitdata1, col = goals, into = c("goal1", "goal2"), sep = ";")


cleanfit$goal2[is.na(cleanfit$goal2)] <- "None"

as.factor(cleanfit$goal2)
