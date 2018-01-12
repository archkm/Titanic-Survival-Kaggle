# Reading the Train and Test Data
Titanic <- read.csv("D:\\PGD Data Science\\Term 2\\Machine Learning\\Kaggle\\Titanic Prediction\\Titanic_Train.csv", 
                    header = TRUE)
Titanic.Kaggle <- read.csv("D:\\PGD Data Science\\Term 2\\Machine Learning\\Kaggle\\Titanic Prediction\\Titanic_Test.csv",
                           header = TRUE)

# Packages
library(e1071)

# Survived As Factor
Titanic$Survived <- as.factor(Titanic$Survived)

# Sampling
Sample.Titanic <- sample.int(n = nrow(Titanic), size = floor(0.7*nrow(Titanic)), 
                             replace = FALSE)

# Train and Test Data
Titanic.Train <- Titanic[Sample.Titanic,]
Titanic.Test <- Titanic[-Sample.Titanic,]

# Test Without Labels
Titanic.Train.Features <- Titanic.Train[,-2]

# Test Without Labels
Titanic.Test.Features <- Titanic.Test[,-2]

# Train Labels
Titanic.Train.Labels <- Titanic.Train[,2]

# Test Labels
Titanic.Test.Labels <- Titanic.Test[,2]

### Decision Tree Model ###
# Packages
library(tree)

# Decision Tree Model 1
Titanic_DT_1 <- tree(Survived ~ Sex + Age, data = Titanic.Train)

# Decision Tree 1 Prediction
Titanic_DT_1.Prediction <- predict(Titanic_DT_1, Titanic.Test, type = "class")
Titanic_DT_1.Prediction
Titanic.Test$DT1_Prediction <- Titanic_DT_1.Prediction

# Confusion Matrix DT 1
Titanic_DT1.ConfMat <- table(Titanic_DT_1.Prediction, 
                            reference = Titanic.Test$Survived)
Titanic_DT1.ConfMat

# Accuracy of DT 1 Model
Titanic_DT1.Accuracy <- sum(diag(Titanic_DT1.ConfMat))/sum(Titanic_DT1.ConfMat)
Titanic_DT1.Accuracy

# Decision Tree Model 2
Titanic_DT_2 <- tree(Survived ~ Pclass + Sex + Age, data = Titanic.Train)

# Decision Tree 2 Prediction
Titanic_DT_2.Prediction <- predict(Titanic_DT_2, Titanic.Test,type = "class")
Titanic_DT_2.Prediction
Titanic.Test$DT2_Prediction <- Titanic_DT_2.Prediction

# Confusion Matrix DT 2
Titanic_DT2.ConfMat <- table(Titanic_DT_2.Prediction, 
                             reference = Titanic.Test$Survived)
Titanic_DT2.ConfMat

# Accuracy of DT 1 Model
Titanic_DT2.Accuracy <- sum(diag(Titanic_DT2.ConfMat))/sum(Titanic_DT2.ConfMat)
Titanic_DT2.Accuracy

### Naive Bayes Model###
Titanic_NB <- naiveBayes(Survived ~ ., data = Titanic.Train)

# Predicting Naive Bayes Model on the Test Data
Titanic_NB.Prediction <- predict(Titanic_NB, Titanic.Test.Features, type = "class")
Titanic.Test$NB_Prediction <- Titanic_NB.Prediction

# Naive Bayes Confusion Matrix
Titanic_NB.ConfMat <- table(Titanic_NB.Prediction, Titanic.Test.Labels)
Titanic_NB.ConfMat

# Accuracy of the Naive Bayes Model
Titanic_NB.Accuracy <- sum(diag(Titanic_NB.ConfMat))/sum(Titanic_NB.ConfMat)
Titanic_NB.Accuracy

# Taking Majority of Predictions
Titanic.Test$PredictionMajority <- as.factor(ifelse(Titanic_DT_1.Prediction == 1 & 
                                                      Titanic_DT_2.Prediction == 1 & 
                                                      Titanic_NB.Prediction == 1, 1, ifelse(Titanic_DT_1.Prediction == 1 & 
                                                                                              Titanic_DT_2.Prediction == 1 & 
                                                                                              Titanic_NB.Prediction == 0, 1, ifelse(Titanic_DT_1.Prediction == 1 & 
                                                                                                                                      Titanic_DT_2.Prediction == 0 & 
                                                                                                                                      Titanic_NB.Prediction == 1, 1, ifelse(Titanic_DT_1.Prediction == 1 & 
                                                                                                                                                                              Titanic_DT_2.Prediction == 0 & 
                                                                                                                                                                              Titanic_NB.Prediction == 0, 0, ifelse(Titanic_DT_1.Prediction == 0 & 
                                                                                                                                                                                                                      Titanic_DT_2.Prediction == 0 & 
                                                                                                                                                                                                                      Titanic_NB.Prediction == 0, 0, ifelse(Titanic_DT_1.Prediction == 0 & 
                                                                                                                                                                                                                                                              Titanic_DT_2.Prediction == 0 & 
                                                                                                                                                                                                                                                              Titanic_NB.Prediction == 1, 0, ifelse(Titanic_DT_1.Prediction == 0 & 
                                                                                                                                                                                                                                                                                                      Titanic_DT_2.Prediction == 1 & 
                                                                                                                                                                                                                                                                                                      Titanic_NB.Prediction == 0, 0, ifelse(Titanic_DT_1.Prediction == 0 & 
                                                                                                                                                                                                                                                                                                                                              Titanic_DT_2.Prediction == 1 & 
                                                                                                                                                                                                                                                                                                                                              Titanic_NB.Prediction == 1, 1, 0)))))))))

# Majority Values into a variable
Titanic.TrainMajority <- Titanic.Test$PredictionMajority

# Modelling for the whole data (Test + Train of the Train File)
# Decision Tree 1
Titanic.Model_DT1 <- tree(Survived ~ Sex + Age, data = Titanic)

# Decision Tree 2
Titanic.Model_DT2 <- tree(Survived ~ Pclass + Sex + Age, data = Titanic)

# Naive Bayes
Titanic.Model_NB <- naiveBayes(Survived ~ ., data = Titanic)

# Predicting for the Test File
# Decision Tree 1
Titanic.Prediction_DT1 <- predict(Titanic.Model_DT1, Titanic.Kaggle, type = "class")

# Decision Tree 2
Titanic.Prediction_DT2 <- predict(Titanic.Model_DT2, Titanic.Kaggle, type = "class")

# Naive Bayes
Titanic.Prediction_NB <- predict(Titanic.Model_NB, Titanic.Kaggle, type = "class")

# Saving the three predictions into variables
Titanic.Kaggle$Model1_DT <- Titanic.Prediction_DT1
Titanic.Kaggle$Model2_DT <- Titanic.Prediction_DT2
Titanic.Kaggle$Model3_NB <- Titanic.Prediction_NB

# Majority rule for the Test File
Titanic.Kaggle$MajorityPrediction <- (ifelse(Titanic.Prediction_DT1 == 1 & 
                                                        Titanic.Prediction_DT2 == 1 & 
                                                        Titanic.Prediction_NB == 1, 1, ifelse(Titanic.Prediction_DT1 == 1 & 
                                                                                                Titanic.Prediction_DT2 == 1 & 
                                                                                                Titanic.Prediction_NB == 0, 1, ifelse(Titanic.Prediction_DT1 == 1 & 
                                                                                                                                        Titanic.Prediction_DT2 == 0 & 
                                                                                                                                        Titanic.Prediction_NB == 1, 1, ifelse(Titanic.Prediction_DT1 == 1 & 
                                                                                                                                                                                Titanic.Prediction_DT2 == 0 & 
                                                                                                                                                                                Titanic.Prediction_NB == 0, 0, ifelse(Titanic.Prediction_DT1 == 0 & 
                                                                                                                                                                                                                        Titanic.Prediction_DT2 == 0 & 
                                                                                                                                                                                                                        Titanic.Prediction_NB == 0, 0, ifelse(Titanic.Prediction_DT1 == 0 & 
                                                                                                                                                                                                                                                                Titanic.Prediction_DT2 == 0 & 
                                                                                                                                                                                                                                                                Titanic.Prediction_NB == 1, 0, ifelse(Titanic.Prediction_DT1 == 0 & 
                                                                                                                                                                                                                                                                                                        Titanic.Prediction_DT2 == 1 & 
                                                                                                                                                                                                                                                                                                        Titanic.Prediction_NB == 0, 0, ifelse(Titanic.Prediction_DT1 == 0 & 
                                                                                                                                                                                                                                                                                                                                                Titanic.Prediction_DT2 == 1 & 
                                                                                                                                                                                                                                                                                                                                                Titanic.Prediction_NB == 1, 1, 0)))))))))

# New Data Frame
Final.Prediction <- Titanic.Kaggle[,c(1,15)]

# Writing as CSV File
write.csv(Final.Prediction, file = "Titanic_Ensemble_Prediction.csv", row.names = FALSE)
