#libraries installation
install.packages('pastecs')
install.packages('readr')
install.packages('Jmisc')
install.packages('ggplot2')
install.packages('plotly')
install.packages('corrplot')
install.packages('caret')
install.packages('e1071')
install.packages('gbm')
install.packages("klaR", dependencies = TRUE)
install.packages("car", dependencies = TRUE)
install.packages("mctest")
install.packages("GGally")
install.packages("caTools")
install.packages('mlbench')
rpart.plot(deciTreeMod, extra = 106)


# libraries used
# for basic data summary
library(pastecs)
library(readr)

# for visualization
library(ggplot2)
library(plotly)
library(corrplot)
library(caTools)

# for checking multicollinearity
library(mctest)
library(car)
library(GGally)

# for applying machine learning algorithms
# for regression & classification
library(caret)
# for svm
library(e1071)
# gradient boast
library(gbm)
# for classification and visualization(knn, k-modes)
library(klaR)
# for decision tree and plotting decision tree
library(rpart)
library(rpart.plot)
# for benchmarking
library(mlbench)

# reading and previewing first 5 rows of the dataset
df <- read.csv("../Churn_Modelling.csv")
head(df, 5)

# checking the information of dataset and number of missing value on each column 
str(df)
sapply(df, function(x) sum(is.na(x)))

# checking the unique values for each column
sapply(df, function(x) length(unique(x)))

# removing first 3 coulumns(features) as they are specific to the customers
df <- subset(df, select = -c(RowNumber, CustomerId,  Surname))
dim(df)
head(df, 2)

# Understanding the y feature of the dataset
# bar-graph
ggplot(df, aes(x = Exited)) + 
  geom_histogram(binwidth = 1, fill = c("blue","Dark Red")) +
  xlab("Exited") +
  ylab("Frequency") +
  ggtitle("Individual who Exited")

# pie-chart
slices <- table(df$Exited)
pct <- round(slices/sum(slices)*100)
lbls<-paste(names(slices),pct,sep=" or ")
lbls<-paste(lbls,"%")
pie(slices, labels=lbls, col=terrain.colors(length(lbls)),
main="Pie Chart of Exited")

# Insight:-
# only 20% of the customers have currently exited and the goal to model 
# this feature, more exploration on the data to see the corelation between 
# some predictors and the outcome variables

# Visualizing how the exited individuals based on tenure
ggplot(df, aes(x = Exited)) +
  geom_histogram(aes(x = Tenure,
                     fill = Exited), binwidth = 1) +
  geom_vline(aes(xintercept=mean(Tenure))) +
  facet_grid( ~ Exited)

# Insight:-
# From the histogram, we can see that average tenure value is around 5 years, 
# for both exited and not exited individual.

# Visualizing how the exited individuals based on gender
ggplot(df,aes(x=Gender, y=Exited)) +
  geom_point(position=position_jitter(0.3))

# Insight:-
# Considering the plot, we can easily see that Female individuals have exited 
# more than Male individuals

# Visualizing how the exited individuals based on Geography
ggplot(df,aes(x=Exited, fill=Geography)) +
  geom_density(col=NA,alpha=0.25)

# Insight:-
# From the visualization, we can say that the individuals living in Germany 
# exited more than those in living in France and Spain.

# Visualizing the correlation between the numerical variables
p1 <- df[,-which(names(df) == "Geography")]
p2 <- p1[,-which(names(p1) == "Gender")]
corr <- cor(p2)
d <- corrplot(corr)

# Converting Categorical Variables to Numerical and Plotting the new correlation
# Converting Categorical Variable (Gender) to Numerical
df$Gender <- as.factor(df$Gender)
df$Gender <- as.numeric(df$Gender) - 1
is.numeric(df$Gender)

# Converting Categorical Variable (Geography) to Numerical
df$Geography <- as.factor(df$Geography)
df$Geography <- as.numeric(df$Geography) - 1
is.numeric(df$Geography)

corr <- cor(df)
d <- corrplot(corr, method="color", order="hclust")

# Insight:-
# From the correlation plot, we can extract the following insights:-
# - Age and Balance has some positive correlation with Exited.
# - Gender and IsActiveMember has some negative correlation with Exited.


# After removing and feature engineering, we have the final summary of the 
# provided dataset
summary(df)
str(df)

# splitting the data into Training & Test set
dataset <- createDataPartition(df$Exited, p = 4/5, list = FALSE)
df_train <- df[dataset,]
df_test <- df[-dataset,]

dim(df_train)
dim(df_test)

# Feature Scaling
# df_train[c(1,4,5,6,7,10)] <- lapply(df_train[c(1,4,5,6,10)], function(x) c(scale(x)))
# df_test[c(1,4,5,6,7,10)] <- lapply(df_test[c(1,4,5,6,10)], function(x) c(scale(x)))

# Build the logistic regression model
logitMod <- glm(Exited ~ ., data=df_train, family=binomial(link="logit"))
summary(logitMod)
# Checking Multicollinearity
vif(logitMod)
plot(logitMod)

# Insight:
# Considering the rule of thumb, if VIF is: 
# 1) 1 = not correlated.
# 2) Between 1 and 5 = moderately correlated.
# We can safely say that the features are not correlated.

# Data Preparation
# Feature Selection: considering relevant and important attributes in your data
# Ranking the features based on importance
set.seed(100)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(Exited~., data=df_train, method="rpart", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

# Insight:-
# Based on the graph, we have see that:-
# - Features like NumOfProducts, Age, IsActive Member, Balance, Gender and
#   Geography are first 6 important feature
# - Other features like EstimatedSalary, CreditScore can be neglected.

################################################################################

# Using the above Logistic Regression Model
logitMod <- glm(Exited ~ NumOfProducts + Age + IsActiveMember + Balance + 
                  Gender + Geography, data=df_train, family=binomial)

logPredict <- predict(logitMod, df_test, type="response")

# ROC Curve
model_AUC <- colAUC(logPredict, df_test$Exited, plotROC = T)
abline(h = model_AUC, col = "Red")
text(.2, .9, cex = .8, labels = paste("Optimal Cutoff:", round(model_AUC,4)))

# Cutoff from the ROC Curve = 0.7319
result_class <- ifelse(logPredict > 0.7319, 1, 0)

# Changing the Exited to binary feature
result_class <- factor(result_class)
actual_class <- factor(df_test$Exited)

# Confusion Matrix
confusionMatrix(result_class, actual_class, mode = "prec_recall", positive="1")

# Insight:-
# The accuracy for the logistic regression model is 79.5%
# From the confusion Matrix, we can see that:-
#   1. Precision : 0.70588         
#   2. Recall : 0.02878         
#   3. F1 : 0.05530

################################################################################

# Using Decision Tree
deciTreeMod <- rpart(Exited ~ NumOfProducts + Age + IsActiveMember + Balance + 
                       Gender + Geography, data=df_train, method = 'class')

rpart.plot(deciTreeMod, extra = 106)

decitreePredict <- predict(deciTreeMod, df_test, type = "class")

levels(decitreePredict)
actual_class <- factor(df_test$Exited)

# Confusion Matrix
confusionMatrix(decitreePredict, actual_class, mode = "prec_recall", positive="1")

# Insight:-
# The accuracy for the logistic regression model is 84.1%
# From the confusion Matrix, we can see that:-
#   1. Precision : 0.7143
#   2. Recall : 0.3957          
#   3. F1 : 0.5093

################################################################################

# Model Comparison
# The main focus is to compare models based on the accuracy. The best model is 
# Decision Tree with 84.1%, followed by Logistic Regression with 79.1%.
# I should also consider other algorithms as well for example: Random Forest,
# Naive Bayes and SVC.

################################################################################

# Conclusion
# - From the descriptive analysis performed, Female individual exited more 
#   than Male and Customers located in Germany churn more comparing to 
#   other locations.
# - With the Prediction made by the models, Decision Tree seems to classify 
#   compared to other model with the accuracy of 84.1%.
# - This accuracy can further be improved by using different algorithms and if 
#   we are able to obtain additional information for the analysis so that we 
#   can improve the feature engineering process.
# - My personal thought is we should concentrate on the individuals who are
#   exited rather than retaining the new individuals. It's always easier to 
#   retain the individual who are with us than obtaining new ones. 

################################################################################