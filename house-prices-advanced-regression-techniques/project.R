library('tidyverse')
library('caret')
library('randomForest')
library('rpart')
library('e1071')

data_folder <- 'house-prices-advanced-regression-techniques/'

# reading train and test csv files
traindf <- read.csv(file.path(data_folder, 'train.csv'), stringsAsFactors = F)
testdf <- read.csv(file.path(data_folder, 'test.csv'), stringsAsFactors = F)

# combine two datasets
alldata <- bind_rows(traindf, testdf)

head(alldata)

# summary of feature data type
data_type <- alldata %>% sapply(class)
data.frame(columns = names(data_type), type = as.character(data_type)) %>% 
  count(type) %>% ggplot(aes(x=type, y=n)) + geom_bar(stat='identity') + 
  xlab('Data Type') + ylab('Number of Features') + coord_flip()
#  43 categorical (factor, character) features
#  38 numeric features

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
alldata <- alldata %>% select(-PoolQC, -MiscFeature, -Alley, -Fence)


# remove utilities column because all is unique among all rows
alldata %>% count(Utilities)
alldata <- alldata %>% select(-Utilities)


# number of categories with categorical columns
unique_values <- alldata %>% keep(is.character) %>% apply(2, function(x) length(unique(x)))
data.frame(features = names(unique_values), vals = as.numeric(unique_values)) %>%
  arrange(-unique_values) %>% 
  ggplot(aes(x=reorder(features, vals), y=vals)) + geom_bar(stat='identity') +
  xlab('Categorical Columns') + ylab('Number of Categories') + coord_flip()


other_ext <- c('CBlock', 'AsphShn', 'BrkComm', 'Stone')

# combine less common factors
# less common factor can create problem druing cross validation
alldata <- alldata %>% mutate(Condition2 = ifelse(Condition2 == 'Norm', 1, 0),
                              Condition1 = ifelse(Condition1 == 'RRNe', 'RRAe', Condition1),
                              Condition1 = ifelse(Condition1 == 'RRNn', 'RRAn', Condition1),
                              ExterCond = ifelse(ExterCond %in% c('Ex', 'Fa', 'Po'), 'Other', ExterCond),
                              RoofMatl = ifelse(RoofMatl == 'CompShg', 'CompShg', 'Other'),
                              Exterior1st = ifelse(Exterior1st == 'ImStucc', 'Stucco', Exterior1st),
                              Exterior1st = ifelse(Exterior1st %in% other_ext, 'Other', Exterior1st),
                              Functional = ifelse(Functional %in% c('Mod', 'Sev'), 'Other', Functional),
                              Heating = ifelse(Heating == 'GasA', 'GasA', 'Other'),
                              HeatingQC = ifelse(HeatingQC %in% c('Po', 'Fa'), 'Other', HeatingQC),
                              GarageCond = ifelse(GarageCond == 'TA', 'TA', 'Other'))



# this function find the most common value in a categorical column
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


# separate train and test data  
traindf <- alldata %>% slice(1:nrow(traindf))
testdf <- alldata %>% slice((nrow(traindf)+1):nrow(alldata))

# Histogram of sale price
traindf %>% ggplot(aes(x=SalePrice/1e3)) + 
  geom_histogram(alpha = 0.7, bins=40, fill='skyblue') + 
  xlab('Sale Price (k$)') + ylab('Histogram')

# Histogram of log sale price
traindf %>% ggplot(aes(x=log(SalePrice/1e3+1))) + 
  geom_histogram(alpha = 0.7, bins=40, fill='skyblue') + 
  xlab('Log(Sale Price) (k$)') + ylab('Histogram')

# top correlation values
traindf %>% keep(is.numeric) %>% cor() %>% data.frame() %>% select(SalePrice) %>%
  mutate(Features = row.names(.)) %>%
  arrange(-abs(SalePrice)) %>%
  drop_na() %>%
  filter(Features != 'SalePrice') %>%
  ggplot(aes(x=reorder(Features, SalePrice), y=SalePrice)) + geom_bar(stat='identity') + 
  xlab('Feature') + ylab('Correlation With SalePrice') + coord_flip()

# relationship between SalePrice and Living Area
traindf %>% ggplot(aes(x=GrLivArea, y=SalePrice/1E3, color=OverallQual)) + geom_point(alpha=0.3) + 
  xlab('Living Area (sqft)') + ylab('Sale Price (k$)')


# relationship between SalePrice and Quality
traindf %>% 
  mutate(OverallQual = factor(OverallQual)) %>%
  ggplot(aes(x=OverallQual, y=SalePrice/1E3, color=OverallQual)) + geom_boxplot() + 
  xlab('Overall Quality') + ylab('SalePrice (k$)')

# relationship between year built and SalePrice
traindf %>% 
  group_by(YearBuilt) %>%
  summarise(SalePrice = mean(SalePrice)) %>%
  ggplot(aes(x=YearBuilt, y=SalePrice/1E3)) + geom_line(color='black') + 
  geom_point(color='darkblue', alpha=0.5) + 
  xlab('Year Built') + ylab('SalePrice (k$)') + geom_smooth(method = 'lm')

# relationship between Garage Area, Number Cars per Garage, Garage Type and Sale Price
traindf %>% ggplot(aes(x=GarageArea, y=SalePrice/1E3, size=GarageCars, color=GarageType)) + 
  geom_point(alpha=0.3) + 
  xlab('Garage Area (sqft)') + ylab('SalePrice (k$)')

# relationship between bathroom and average SalePrice
traindf %>% group_by(FullBath, HalfBath) %>% summarise(SalePrice = mean(SalePrice)) %>%
  ggplot(aes(x=FullBath, y=HalfBath)) + geom_tile(aes(fill = SalePrice/1e3), colour = "white") + 
  scale_fill_gradient(low = "white", high = "steelblue")


## Modeling 
# create a linear regression model with all predictors
full_model <- lm(log(SalePrice) ~ . -Id, data = traindf)

# use step function to find the optimal set of predictors (feature selection)
best_model <- step(full_model, direction = 'backward', trace = 1)
summary(best_model)

# get the best set of predictors
best_formula <- best_model$call$formula

# split the data into 5 groups for kfold cross-validation
set.seed(1206)
nfold <- 10
test_indx <- createFolds(traindf$SalePrice, k = nfold)

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


rmse_vals <- matrix(0, nrow = nfold, ncol = 4)
r2_vals <- matrix(0, nrow = nfold, ncol = 4)

# 5fold cross_validation, no hyperparameter tuning
# random Forest (mtry=p/3, p = 75 number of features), number of trees is 20, and rest paramteres are set to default
# default for ntree = 500 but it is time consuming 
# for SVM (kernel = radial, cost = 1) and CART (maxdepth = 30, cp=0.01), default paramteres are selected

# out of 75 features, 48 features are selected using Step function. 
# we used only selected 48 features for LM, CART, and SVM.
# for random forest we used all 75 features because RF has its own feature selection

k <- 1
for (fold in test_indx){
  
  # train the model
  model1 <- lm(best_formula, data=traindf[-fold, ]) # linear regression (LR)
  model2 <- randomForest(log(SalePrice) ~ . -Id, data=traindf[-fold, ], ntree=20) # random forest (RF)
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
  ggplot(aes(x=model, y=mean)) + geom_bar(stat='identity') + 
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


model_rf <- randomForest(log(SalePrice) ~ . -Id, data=traindf, ntree=20)
model_rf$importance %>% data.frame() %>% mutate(Feature = row.names(.)) %>%
  arrange(-IncNodePurity) %>% head(40) %>% 
  ggplot(aes(x=reorder(Feature,IncNodePurity), y=IncNodePurity, fill=IncNodePurity)) + geom_bar(stat='identity') + coord_flip() + 
  xlab('Feature') + ylab('Importance') + theme(legend.position = "none")
