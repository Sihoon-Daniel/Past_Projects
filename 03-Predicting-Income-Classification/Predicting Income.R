#installing and using package
install.packages(c("data.table", "ggplot2", "ggthemes", "scales", "rpart", "randomForest",
                   "glmnet","gbm", "rpart.plot","fastDummies","tidyverse","scorecard"))
library(fastDummies)
library(tidyverse)
library(randomForest)
library(scorecard)
library(data.table)
library(caret)
library(glmnet)
library(ggplot2)
library(ggthemes)
library(scales)
library(rpart)
library(gbm)
library(rpart.plot)


#Reading the data and saving it to a variable named df
df <- fread("adult_revised.csv",stringsAsFactors = TRUE)


#Creating dummy variables, removing the first column and selected columns
dd <- dummy_cols(df, select_columns = c('Employment', 'Education', 'Marital_status','Occupation', 'Relationship', 'Race', 'Sex'), 
                      remove_first_dummy = TRUE, remove_selected_columns = TRUE)

#Setting US as 1 and others as 0
dd$US<-ifelse(dd$Country == 'United-States',1,0)
dd$Country <- NULL

#rearranges column order so that income is placed in the last row
dd <- dd[,c(1,2,3,4,5,6,8:58,7)]

#Converting the categorical values into 0 and 1
#dd$Income[dd$Income == "<=50K"] <- 0
#dd$Income[dd$Income == ">50K"] <- 1
names(dd) <- make.names(names(dd))
head(dd,20)

set.seed(1)
id <- createDataPartition(dd$Income, p = 0.8, list = FALSE)
train<-dd[id, ]
test<-dd[-id, ]
formula <- as.formula(Income ~ .)

train2 <- train
test2 <- test
test2$Income <- ifelse(test2$Income == "<=50K", 0, 1)
train2$Income <- ifelse(train2$Income == "<=50K", 0, 1)
y_train <- train2$Income
y_test <- test2$Income
X_train <- model.matrix(as.formula(Income~.),train)[,-1]
X_test <-  model.matrix(as.formula(Income~.),test)[,-1]

#Initialize instance of random forest
#randomForest(Income ~., data = train, do.trace=T, mtry = 10)

#Base Model
randomForest(formula, data = train, do.trace=T, mtry=10, keep.inbag=TRUE)

#Bagging Model, mtry = 57 because the last one is Income
randomForest(formula, data = train2, do.trace=T, mtry=57, keep.inbag=TRUE)


fit.rndfor <- randomForest(formula, data = train2, mtry=10, do.trace=T)
varImpPlot(fit.rndfor)
 yhat.rndfor <- predict(fit.rndfor, train)

mse.tree <- mean((yhat.rndfor - y_train) ^ 2)
print(mse.tree)

#Correlation Heatmap
ddf <- dd
ddf$Income <- NULL
cormat <- round(cor(ddf),2)
install.packages("reshape2")
library(reshape2)
melted_cormat <- melt(cormat)
head(melted_cormat)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


#Boosting
# set d to 2 or 3
# set the lambda to 0.1 , and do CV

fit.btree <- gbm(formula, data = train2,
                 distribution = "gaussian",
                 n.trees = 100,
                 interaction.depth = 2,
                 shrinkage = 0.001)

relative.influence(fit.btree)

fit.btree
y_train

yhat.btree <- predict(fit.btree, train, n.trees = 100)
mse.btree <- mean((yhat.btree - y_train) ^ 2)
print(mse.btree)




ggplot(dd, aes(x = Age, y = Years_of_education, color = Income))+geom_point()

ggplot(dd, aes(x = Capital_Gain, fill = Income))
#  geom_histogram(binwidth = .1,
#                 center = 0.05)

ggplot(dd, aes(x = Age, y = Years_of_education, color = Income))+geom_point(position = "jitter")+
  labs(x = "Age", y = "Years of Education", color = "Income")




#lasso
fit_lasso <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1, nfolds = 10)
## Predict responses
y_train_hat <-predict(fit_lasso,newx=X_train,type='response')
y_test_hat <-predict(fit_lasso,newx=X_test,type='response')
mse.min <- fit_lasso$cvm[which(fit_lasso$lambda == fit_lasso$lambda.min)]
print(mse.min)


## Compute MSEs
###lasso
mse_test_las <-colMeans((y_test-y_test_hat)^2)
mse_train_las <-colMeans((y_train-y_train_hat)^2)
print(mse_test_las)
print(mse_train_las)

min_mse_test_las <- fit_lasso$cvm[mse_test_las]
min_mse_train_las<- fit_lasso$cvm[mse_train_las]
print(min_mse_test_las)
print(min_mse_train_las)

fit_lasso

lambda_min_mse_train_las <- fit_lasso$lambda[which.min(mse_train_las)]
lambda_min_mse_test_las <- fit_lasso$lambda[which.min(mse_test_las)]

print(lambda_min_mse_train_las)
print(lambda_min_mse_test_las)

mse_las <- data.table(lambda = fit_lasso$lambda,mse = mse_train_las,dataset = "Train")
mse_las <- rbind(mse_las, data.table(lambda = fit_lasso$lambda,mse = mse_test_las,dataset = "Test"))
print(mse_las)

min_mse_las<-fit_lasso$lambda.min
print(min_mse_las)
print(log(min_mse_las))

plot(fit_lasso)

dd_mse <- data.table(lambda = log(fit_lasso$lambda), mse = fit_lasso$cvm)
ggplot(dd_mse, aes(lambda, mse)) + geom_line() + 
  geom_point(aes(x=log(fit_lasso$lambda.min),y=fit_lasso$cvm[which(fit_lasso$lambda == fit_lasso$lambda.min)],colour='red',size=5)) +
  scale_y_continuous("MSE") + scale_x_continuous("Log(lambda)")



#Ridge

fit_ridge <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 0, nfolds = 10)
## Predict responses
y_train_rig <-predict(fit_ridge,newx=X_train,type='response')
y_test_rig <-predict(fit_ridge,newx=X_test,type='response')
mse.min_rid <- fit_ridge$cvm[which(fit_ridge$lambda == fit_ridge$lambda.min)]
print(mse.min_rid)

mse_test_rid <-colMeans((y_test-y_train_rig)^2)
mse_train_rid <-colMeans((y_train-y_train_rig)^2)
print(mse_test_rid)
print(mse_train_rid)

lambda_train_rid <- mse_train_rid[which.min(mse_train_rid)]
lambda_test_rid <- mse_test_rid[which.min(mse_test_rid)]
print(lambda_train_rid)
print(lambda_test_rid)

mse_rid <- data.table(lambda = fit_ridge$lambda,mse = mse_train_rid,dataset = "Train")
mse_rid <- rbind(mse_las, data.table(lambda = fit_ridge$lambda,mse = mse_test_rid,dataset = "Test"))
print(mse_rid)

min_mse_rid<-fit_ridge$lambda.min
print(min_mse_rid)
print(log(min_mse_rid))

plot(fit_ridge)

dd_mse_rid <- data.table(log_lambda = log(fit_ridge$lambda), mse = fit_ridge$cvm)
ggplot(dd_mse_rid, aes(log_lambda, mse)) + geom_line() + geom_point(aes(x=log(fit_ridge$lambda.min),y=fit_ridge$cvm[which(fit_ridge$lambda == fit_ridge$lambda.min)],colour='red',size=5)) + scale_y_continuous("MSE")


fit.lm <- lm(formula, train2)
yhat.train.lm <- predict(fit.lm)
mse.train.lm <- mean((y_train - yhat.train.lm)^2)
print(mse.train.lm)

yhat.train.lm



head(X_train)





