setwd("data")
getwd()

library(readxl)
library(tm)
library(SnowballC)
library(caTools)
library(e1071)

data <- read.csv("spam_ham_dataset.csv")

data$X <- NULL
data$label <- as.factor(data$label)

data <- data[1:500,]

labels <- data$label
emails <- data$text

corpus <- VCorpus(VectorSource(emails))
corpus <- tm_map(corpus,content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument, language = "en")

termDocMatrix <- TermDocumentMatrix(corpus)

docTermMatrix <- as.DocumentTermMatrix(termDocMatrix)

data <- as.data.frame.matrix(docTermMatrix)
data$label <- labels

sample <- sample.split(data$label, SplitRatio = 0.70)

train_data <- subset(data, sample == TRUE)
test_data <- subset(data, sample == FALSE)

model_nb <- naiveBayes(label ~ ., data = train_data)

predicoes <- predict(model_nb, newdata = test_data)

matriz_confusao <- table(predicoes, test_data$label)
print(matriz_confusao)

