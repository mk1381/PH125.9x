#Load libraries
library(MASS)
library(caret)
library(corrplot)
library(KernSmooth)

#Let's include a stratified holdout function for future use
stratified_holdout <- function(y, epsilon){
  n <- length(y)
  labels <- unique(y)
  ind_train <- NULL
  ind_test <- NULL
  y <- sample(sample(sample(y))) #Resample y multiple times
  for(label in labels){ #Go through each label
    labels_i <- which(y == label)
    n_i <- length(labels_i)
    ind_train <- c(ind_train, sample(labels_i, 
                                     round((1 - epsilon)*n_i), replace = FALSE)) #Select (1 - epsilon)% for train
  }
  ind_test <- (1:n)[-ind_train] #Everything not in training set is in test set
  return(list(ind_train, ind_test))
}

#Load the data
xy <- read.csv('bananaData.csv')

#Shape things nicely for data computation and scale data
n <- nrow(xy) #Number of examples
p <- ncol(xy) - 1 #Number of predictors
pos_resp <- p + 1 #Position of response
x <- scale(xy[,-pos_resp]) #Scaled predictor variables
xy[,pos_resp] <- as.factor(xy[,pos_resp]) #Turn response to factor
y <- xy[,pos_resp] #Response variable

#Explore the data
str(xy) #Data Types
head(xy) #Get an idea of the data
which(is.na(xy)) #Check for empties
plot(xy[,-pos_resp], col = as.integer(y) + 1L, pch = as.integer(y) + 1L, main = "Banana Dataset")

#Check correlation between predictors
corrplot(cor(x))

#Plot the data with color representing class, y = 1 is in green, y = 0 is in red
plot(x, col = as.integer(y) + 1L, pch = as.integer(y) + 1L, main = "Banana Dataset")

#Determine optimal hyperparameters for the kNN, SVM (linear and RBF), 
#decision tree, random forest, and XGBoost models

#Cross - validation control
cv_control <- trainControl(method = "cv", number = 10)

#Optimal k in kNN
mod_knn <- train(x, y, method = "knn", 
                 tuneGrid = expand.grid(k = 1:25), 
                 trControl = cv_control) #kNN, CV for k = 1 ... 25

k_knn_opt <- mod_knn$bestTune$k

#Optimal c in linear SVM
mod_lsvm <- train(x, y, method = "svmLinear",
                 tuneGrid = expand.grid(C = 2^(-5:15)),
                 trControl = cv_control)

c_lsvm_opt <- mod_lsvm$bestTune$C

#Optimal C and sigma in RBF SVM
mod_rsvm <- train(x, y, method = "svmRadial",
                  tuneGrid = expand.grid(C = 2^(-5:15), sigma = 2^(-15:3)),
                  trControl = cv_control)

c_rsvm_opt <- mod_rsvm$bestTune$C
s_rsvm_opt <- mod_rsvm$bestTune$sigma

#Optimal depth in classification tree
mod_tree <- train(x, y, method = "rpart2",
                  tuneGrid = expand.grid(maxdepth = 1:12),
                  trControl = cv_control)

d_tree_opt <- mod_tree$bestTune$maxdepth

#Optimal mtry in RF (can only be 1 or 2 for us)
mod_rf <- train(x, y, method = "rf",
                  tuneGrid = expand.grid(mtry = 1:2),
                  trControl = cv_control)

mtry_rf_opt <- mod_rf$bestTune$mtry

#Optimal training rounds, depth, and learning rate in XGBoost
mod_xgb <- train(x, y, method = "xgbTree",
                tuneGrid = expand.grid(nrounds = c(50, 100, 200, 400), 
                                       max_depth = 1:6,
                                       eta = c(0.2, 0.3, 0.5, 0.8),
                                       gamma = 0,
                                       colsample_bytree = 1,
                                       min_child_weight = 1,
                                       subsample = 1),
                trControl = cv_control)

n_xgb_opt <- mod_xgb$bestTune$nrounds
d_xgb_opt <- mod_xgb$bestTune$max_depth
e_xgb_opt <- mod_xgb$bestTune$eta



#Let's perform stochastic holdout to measure the performance of each machine
#over 25 runs

R <- 25 #Number of runs
error_sh <- matrix(0, nrow = R, ncol = 10) #Matrix for storing errors
no_control <- trainControl(method = 'none') #No trainControl since hyperparameters are known

for(r in 1:R){
  
  #Generate random stratified split
  indices <- stratified_holdout(y, 1/3)
  ind_train <- indices[[1]]
  ind_test <- indices[[2]]
  
  #Train models
  mod_lda <- train(x[ind_train,], y[ind_train], method = "lda") #LDA
  mod_qda <- train(x[ind_train,], y[ind_train], method = "qda") #QDA
  mod_nb <- train(x[ind_train,], y[ind_train], method = "nb") #Naive Bayes
  mod_knn <- train(x[ind_train,], y[ind_train], method = "knn", trControl = no_control,
                   tuneGrid = expand.grid(k = k_knn_opt)) #kNN
  mod_lr <- train(x[ind_train,], y[ind_train], method = "glm", family = binomial(link = "logit")) #LR
  mod_lsvm <- train(x[ind_train,], y[ind_train], method = "svmLinear", trControl = no_control,
                    tuneGrid = expand.grid(C = c_lsvm_opt)) #Linear SVM
  mod_rsvm <- train(x[ind_train,], y[ind_train], method = "svmRadial", trControl = no_control,
                    tuneGrid = expand.grid(C = c_rsvm_opt, sigma = s_rsvm_opt)) #RBF SVM
  mod_tree <- train(x[ind_train,], y[ind_train], method = "rpart2", trControl = no_control,
                    tuneGrid = expand.grid(maxdepth = d_tree_opt)) #Tree
  mod_rf <- train(x[ind_train,], y[ind_train], method = "rf", trControl = no_control,
                    tuneGrid = expand.grid(mtry = mtry_rf_opt)) #RF
  mod_xgb <- train(x[ind_train,], y[ind_train], method = "xgbTree", trControl = no_control,
                  tuneGrid = expand.grid(nrounds = n_xgb_opt, 
                                         max_depth = d_xgb_opt,
                                         eta = e_xgb_opt,
                                         gamma = 0,
                                         colsample_bytree = 1,
                                         min_child_weight = 1,
                                         subsample = 1)) #XGBoost
  
  #Make predictions
  y_hat_lda <- predict(mod_lda, x[ind_test,])
  y_hat_qda <- predict(mod_qda, x[ind_test,])
  y_hat_nb <- predict(mod_nb, x[ind_test,])
  y_hat_knn <- predict(mod_knn, x[ind_test,])
  y_hat_lr <- predict(mod_lr, x[ind_test,])
  y_hat_lsvm <- predict(mod_lsvm, x[ind_test,])
  y_hat_rsvm <- predict(mod_rsvm, x[ind_test,])
  y_hat_tree <- predict(mod_tree, x[ind_test,])
  y_hat_rf <- predict(mod_rf, x[ind_test,])
  y_hat_xgb <- predict(mod_xgb, x[ind_test,])
  
  #Store errors
  error_sh[r, 1] <- mean(y_hat_lda != y[ind_test])
  error_sh[r, 2] <- mean(y_hat_qda != y[ind_test])
  error_sh[r, 3] <- mean(y_hat_nb != y[ind_test])
  error_sh[r, 4] <- mean(y_hat_knn != y[ind_test])
  error_sh[r, 5] <- mean(y_hat_lr != y[ind_test])
  error_sh[r, 6] <- mean(y_hat_lsvm != y[ind_test])
  error_sh[r, 7] <- mean(y_hat_rsvm != y[ind_test])
  error_sh[r, 8] <- mean(y_hat_tree != y[ind_test])
  error_sh[r, 9] <- mean(y_hat_rf != y[ind_test])
  error_sh[r, 10] <- mean(y_hat_xgb != y[ind_test])
}

#Names of models in order
mod_names <- c("LDA", "QDA", "NB", "kNN", "LR", "SVM", "RBF SVM", "Tree", "RF", "XGB")

#Error means
err_means <- data.frame(mod_names, colMeans(error_sh))
err_means

#Generate a boxplot of the errors
boxplot(error_sh, main = "Boxplots of Performance on Banana Dataset by Various Models over 25 Runs", 
        names = mod_names, 
        ylab = "Avg Error")
