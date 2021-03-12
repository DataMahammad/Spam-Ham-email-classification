library(data.table)
library(tidyverse)


dataset <- fread("emails.csv")

dataset %>% dim()

dataset %>% nrow()

id = seq(1,dataset %>% nrow(),by = 1)

length(id)

dataset$id <- id

dataset %>% colnames()

dataset %>% glimpse()

dataset$id <- dataset$id %>% as.character()

dataset$spam %>% table() %>% prop.table()

#In the future,apply imbalance problem!!!!!!!!

dataset %>% is.null() %>% sum()

library(inspectdf)

dataset %>% inspect_na()

#?sample.split()
library(text2vec)
library(caTools)
library(glmnet)


#Splitting
set.seed(123)
split <- dataset$spam %>% sample.split(SplitRatio = 0.8)
train <- dataset %>% subset(split == T)
test <- dataset %>% subset(split == F)


#Tokenizer
train %>% colnames()

train_tokens <- train$text %>% tolower() %>% word_tokenizer()

it_train <- train_tokens %>% 
  itoken(ids = train$id,
         progressbar = F)

vocab <- it_train %>% create_vocabulary()

vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10)


vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer) #dtm is document-term matrix


glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)# again lower number of iterations for faster training

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

test %>% colnames()

it_test <- test$text %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$id,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)

#train and test AUC are 0.99 --> seems to be no overfitting as low bias and low variance 


