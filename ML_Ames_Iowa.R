library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(e1071)
library(tidyverse)
library(caret)
library(lasso2)
library(ggplot2)
library(dplyr)
library(scales)
library(ggcorrplot)
library(car)
library(data.table)
library(reshape)

#############################################
###########IMPORT#DATA#######################
#############################################

# reading train and test csv files
traindf <- read.csv(file.path(data_folder, 'train.csv'), stringsAsFactors = F)
testdf <- read.csv(file.path(data_folder, 'test.csv'), stringsAsFactors = F)

#################################################
#############DATA#WRANGLING#####################
###############################################


# combine two datasets for imputation and feature engineering
# hold Sale price so it does not get imputed

SalePrice <- traindf$SalePrice
alldata <- rbind(within(traindf, rm('Id', 'SalePrice')), within(testdf, rm('Id')))


alldata
str(alldata)

###data EDA####

data_type <- alldata %>% sapply(class)
data.frame(columns = names(data_type), type = as.character(data_type)) %>% 
  count(type) %>% ggplot(aes(x=type, y=n)) + geom_bar(stat='identity') + 
  xlab('Data Type') + ylab('Number of Features') + coord_flip()

# missing values
missing <- alldata %>% is.na() %>% colMeans()
data.frame(features = names(missing), prctn = as.numeric(missing)) %>%
  filter(features != 'SalePrice', features != 'Id') %>%
  arrange(-prctn) %>% head(40) %>%
  ggplot(aes(x=reorder(features, prctn), y=prctn)) + geom_bar(stat='identity') +
  xlab('Feature') + ylab('Percentage of Missing Values') + coord_flip()
# PoolQC columns has more than 98% missing values
# MiscFeature has more than 90% missing values

# remove columns with too many missing values
alldata$PoolArea <- NULL
alldata$PoolQC <- NULL
alldata$MSSubClass <- NULL
alldata$Alley <- NULL
alldata$Utilities <- NULL
alldata$Fence <- NULL


######Featrue engineering########

# Total square footage
alldata$TotalSF <- alldata$GrLivArea + alldata$TotalBsmtSF
# yard size
alldata$YardSize <- alldata$LotArea-alldata$X1stFlrSF


# Quality - convert to numeric and sum for overall to reduce columns
# scale (sum) to match overall quality from 1-10

alldata$GarageQual <- as.numeric(factor(alldata$GarageQual, levels = c("Ex", "Fa", "Gd", "TA","Po", "NA"), labels = c(5,2,4,3,1,0), ordered = TRUE))
alldata$GarageCond <- as.numeric(factor(alldata$GarageCond, levels = c("Ex", "Fa", "Gd", "TA","Po", "NA"), labels = c(5,2,4,3,1,0), ordered = TRUE))
alldata$GarageOverall <- alldata$GarageQual+alldata$GarageCond
alldata$GarageQual <- NULL
alldata$GarageCond <- NULL

alldata$ExterQual <- as.numeric(factor(alldata$ExterQual, levels = c("Ex", "Fa", "Gd", "TA","Po", "NA"), labels = c(5,2,4,3,1,0), ordered = TRUE))
alldata$ExterCond <- as.numeric(factor(alldata$ExterCond, levels = c("Ex", "Fa", "Gd", "TA","Po", "NA"), labels = c(5,2,4,3,1,0), ordered = TRUE))
alldata$ExteriorOverall <- alldata$ExterQual+alldata$ExterCond
alldata$ExterQual <- NULL
alldata$ExterCond <- NULL

alldata$BsmtQual <- as.numeric(factor(alldata$BsmtQual, levels = c("Ex", "Fa", "Gd", "TA","Po", "NA"), labels = c(5,2,4,3,1,0), ordered = TRUE))
alldata$BsmtCond <- as.numeric(factor(alldata$BsmtCond, levels = c("Ex", "Fa", "Gd", "TA","Po", "NA"), labels = c(5,2,4,3,1,0), ordered = TRUE))
alldata$BsmtOverall <- alldata$BsmtQual+alldata$BsmtCond
alldata$BsmtCond <- NULL
alldata$ExterCond <- NULL

# Quality - to scale with overall
alldata$HeatingQC <- as.numeric(factor(alldata$HeatingQC, levels = c("Ex", "Fa", "Gd", "TA","Po", "NA"), labels = c(9,7,5,3,1,0), ordered = TRUE))
alldata$KitchenQual <- as.numeric(factor(alldata$KitchenQual, levels = c("Ex", "Fa", "Gd", "TA","Po", "NA"), labels = c(9,7,5,3,1,0), ordered = TRUE))
alldata$FireplaceQu <- as.numeric(factor(alldata$FireplaceQu, levels = c("Ex", "Fa", "Gd", "TA","Po", "NA"), labels = c(9,7,5,3,1,0), ordered = TRUE))



str(alldata)

# number of categories with categorical columns
unique_values <- alldata %>% keep(is.character) %>% apply(2, function(x) length(unique(x)))
data.frame(features = names(unique_values), vals = as.numeric(unique_values)) %>%
  arrange(-unique_values) %>% 
  ggplot(aes(x=reorder(features, vals), y=vals)) + geom_bar(stat='identity') +
  xlab('Categorical Columns') + ylab('Number of Categories') + coord_flip()


# combine less common factors
# less common factor can create problem during cross validation
other_ext <- c('CBlock', 'AsphShn', 'BrkComm', 'Stone')
alldata <- alldata %>% mutate(Condition2 = ifelse(Condition2 == 'Norm', 1, 0),
                              Condition1 = ifelse(Condition1 == 'RRNe', 'RRAe', Condition1),
                              Condition1 = ifelse(Condition1 == 'RRNn', 'RRAn', Condition1),
                              RoofMatl = ifelse(RoofMatl == 'CompShg', 'CompShg', 'Other'),
                              Exterior1st = ifelse(Exterior1st == 'ImStucc', 'Stucco', Exterior1st),
                              Exterior1st = ifelse(Exterior1st %in% other_ext, 'Other', Exterior1st),
                              Functional = ifelse(Functional %in% c('Mod', 'Sev'), 'Other', Functional),
                              Heating = ifelse(Heating == 'GasA', 'GasA', 'Other'))

# function to find the most common value in a categorical column
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# replace missing values with most common value for categorical columns
alldata <- alldata %>% mutate_if(is.character, funs(replace(.,is.na(.), Mode(na.omit(.)))))

# replace missing value with the mean for numeric columns
alldata <- alldata %>% mutate_if(is.numeric, funs(replace(.,is.na(.), mean(na.omit(.)))))

# convert character columns to factor for modeling
alldata <- alldata %>% mutate_if(is.character, as.factor)

anyNA(alldata)
#success!

str(alldata)

##########################################################
#########Assumptions and diagnostics######################
##########################################################

# separate train and test data  
traindf <- alldata %>% slice(1:nrow(traindf))
traindf <- traindf %>% cbind(SalePrice)
testdf <- alldata %>% slice((nrow(traindf)+1):nrow(alldata))

summary(traindf)
str(traindf)
#73 Varaibles (not includeing sale price)

###sale price distribution of training data##
# need to log transform sale price so  that it does not 
# violate the assumption of normality for linear regression
ggplot(traindf, aes(x=SalePrice, fill = ..count..)) +
  geom_histogram(fill = 'purple', binwidth = 10000) +
  ggtitle("Sale Prices in Ames, IA from 2006-2010") +
  ylab("Count of Houses Sold") +
  xlab("Housing Price") +
  scale_x_continuous(label = comma)

# Histogram of log sale price
ggplot(traindf, aes(x=log(SalePrice), fill = ..count..)) +
  geom_histogram(fill = 'purple', binwidth = 0.05) +
  ggtitle("Log SalePrices Histogram") +
  scale_x_continuous(label = comma)

ggplot(train, aes(x = SalePrice, fill = as.factor(OverallQual))) +
  geom_histogram(position = "stack", binwidth = 10000) +
  ggtitle("Sales by Overall Quality of House") +
  ylab("Sales") +
  xlab("Housing Price") +
  scale_x_continuous(label=comma) +
  scale_fill_discrete(name="Overall Quality")+
  theme(plot.title = element_text(hjust = 0.5), 
        legend.position=c(0.8,0.6), 
        legend.background = element_rect(fill="grey90",
                                         linetype="solid", 
                                         colour ="black"))

# relationship between SalePrice and Living Area
traindf %>% ggplot(aes(x=TotalSF, y=SalePrice, color=YrSold)) + 
  geom_point(alpha=.5) + 
  xlab('Total square Footage of House') + ylab('Sale Price')+
  scale_y_continuous(label=dollar)

traindf %>% 
  mutate(YrSold = factor(YrSold)) %>%
  ggplot(aes(x=YrSold, y=SalePrice)) + 
  geom_boxplot() + 
  xlab('YearSold') + ylab('SalePrice')+
  scale_y_continuous(label = dollar)

traindf %>% 
  mutate(MoSold = factor(MoSold)) %>%
  ggplot(aes(x=MoSold, y=SalePrice)) + 
  geom_boxplot() + 
  xlab('Month Sold') + ylab('Sale Price')+
  scale_y_continuous(label = dollar)


##############################################
##############Modeling########################
##############################################

# create a linear regression model with all predictors
full_model <- lm(log(SalePrice) ~ ., data = traindf)

# use step function to find the optimal set of predictors (feature selection)
best_model <- step(full_model, direction = 'backward', trace = 1)
summary(best_model)

# get the best set of predictors
best_formula <- best_model$call$formula
best_formula

# examine results
summary(best_model)
plot(best_model)
influencePlot(best_model)
vif(best_model)
avPlots(best_model)
confint(best_model)

####################
###MODEL CHECKING###
####################

###USE CROSS VALIDATION TO CHOOSE BEST MODEL ON TRAINING DATA###

# split the data into 5 groups for kfold cross-validation
set.seed(10118)
test_indx <- createFolds(traindf$SalePrice, k = 5)

# function to calcualte root mean square error (rmse), lower is better
rmse <- function(y_true, y_pred){
  return (sqrt(mean((y_true - y_pred)^2)))
}

# function to calcualte r squared (between 0 and 1), higher is better
r2 <- function(y_true, y_pred){
  sstot <- sum((y_true - mean(y_true))^2)
  ssres <- sum((y_true - y_pred)^2)
  return (1 - ssres/sstot)
}


rmse_vals <- matrix(0, nrow = 5, ncol = 4)
r2_vals <- matrix(0, nrow = 5, ncol = 4)

k <- 1
for (fold in test_indx){
  
  # train the model
  model1 <- lm(best_formula, data=traindf[-fold, ]) # linear regression (LR)
  model2 <- randomForest(log(SalePrice) ~ ., data=traindf[-fold, ], ntree=20) # random forest (RF)
  model3 <- rpart(best_formula, data=traindf[-fold, ]) # decision tree (CART)
  model4 <- svm(best_formula, data=traindf[-fold, ]) # support vector machine (SVM)
  
  # prediction on test set
  y_pred1 <- predict(model1, newdata = traindf[fold,])
  y_pred2 <- predict(model2, newdata = traindf[fold,])
  y_pred3 <- predict(model3, newdata = traindf[fold,])
  y_pred4 <- predict(model4, newdata = traindf[fold,])
  
  # calcualte rmse for each model
  rmse_vals[k,1] <- rmse(traindf$SalePrice[fold], exp(y_pred1))
  rmse_vals[k,2] <- rmse(traindf$SalePrice[fold], exp(y_pred2))
  rmse_vals[k,3] <- rmse(traindf$SalePrice[fold], exp(y_pred3))
  rmse_vals[k,4] <- rmse(traindf$SalePrice[fold], exp(y_pred4))
  
  # calculate r2 for each model
  r2_vals[k,1] <- r2(traindf$SalePrice[fold], exp(y_pred1))
  r2_vals[k,2] <- r2(traindf$SalePrice[fold], exp(y_pred2))
  r2_vals[k,3] <- r2(traindf$SalePrice[fold], exp(y_pred3))
  r2_vals[k,4] <- r2(traindf$SalePrice[fold], exp(y_pred4))
  
  k <- k + 1
  
}

rmse_vals <- data.frame(rmse_vals)
colnames(rmse_vals) <- c('LinearRegression', 'RandomForest', 'CART', 'SVM')

r2_vals <- data.frame(r2_vals)
colnames(r2_vals) <- c('LinearRegression', 'RandomForest', 'CART', 'SVM')


rmse_vals %>% gather('model', 'rmse') %>% group_by(model) %>%
  summarise(mean = mean(rmse), sd = sd(rmse)) %>% 
  ggplot(aes(x=model, y=mean)) + 
  geom_bar(stat='identity') + 
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = .2) + 
  xlab('Model') + ylab('RMSE')


r2_vals %>% gather('model', 'r2') %>% group_by(model) %>%
  summarise(mean = mean(r2), sd = sd(r2)) %>% 
  ggplot(aes(x=model, y=mean)) + geom_bar(stat='identity') + 
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = .2) + 
  xlab('Model') + ylab('R2')

r2_vals %>% gather('model', 'r2') %>%
  cbind(rmse_vals %>% gather('model', 'rmse') %>% select(rmse)) %>%
  group_by(model) %>%
  summarise_all(mean) %>%
  gather('key', 'val', -model) %>%
  ggplot(aes(x=model, y=val)) + geom_bar(stat='identity') + facet_wrap(~key, scales = 'free') + 
  coord_flip()


model_rf <- randomForest(log(SalePrice) ~ ., data=traindf, ntree=20)
model_rf$importance %>% data.frame() %>% mutate(Feature = row.names(.)) %>%
  arrange(-IncNodePurity) %>% head(40) %>% 
  ggplot(aes(x=reorder(Feature,IncNodePurity), y=IncNodePurity, fill=IncNodePurity)) + geom_bar(stat='identity') + coord_flip() + 
  xlab('Feature') + ylab('Importance') + theme(legend.position = "none")

r2_vals
rmse_vals

###########################
#####MODEL BUILDING#######
##########################

### CHOOSE MODEL FROM CROSS VALIDATION ###

### USE MODEL ON TEST SET  FOR PREDICTIONS ###
