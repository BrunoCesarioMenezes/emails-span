setwd("data")

#Certifique-se de ter os seguintes pacotes e suas dependências
#instaladas para rodar o script R
library(readxl)
library(tm)
library(SnowballC)
library(caTools)
library(e1071)

#Lê base de dados, salva os resultados finais em
#labels e salva os emails
data <- read.csv("spam_ham_dataset.csv")

data$X <- NULL
data$label <- as.factor(data$label)

labels <- data$label
emails <- data$text

#Usa base dos emails para tratar no textMining

corpus <- VCorpus(VectorSource(emails))
corpus <- tm_map(corpus,content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument, language = "en")

#Cria matriz de termos-documentos com os emails tratados, depois
#converte para matriz de documentos-termos com peso

termDocMatrix <- TermDocumentMatrix(corpus)
termDocMatrix <- removeSparseTerms(termDocMatrix, 0.85)
tdm_tfidf <- weightTfIdf(termDocMatrix)

docTermMatrix <- as.DocumentTermMatrix(tdm_tfidf)

#Converte matriz de TDM para dataframe e pega amostras de treino e teste
data <- as.data.frame.matrix(docTermMatrix)
data$label <- labels

sample <- sample.split(data$label,0.70)

train_data <- subset(data, sample == TRUE)
test_data <- subset(data, sample == FALSE)

#Treina modelo usando naiveBayes e faz predicoes puras (raw)
model_nb <- naiveBayes(label ~ ., data = train_data)

predicoes <- predict(model_nb, newdata = test_data, type = "raw")

#Trata predicoes dos emails de acordo com a probabilidade
#(deixa critério do modelo mais liberal)

novo_predicoes <- ifelse(predicoes[, "spam"] > 0.9999999999999, "spam", "ham")

novo_predicoes <- as.factor(novo_predicoes)

matriz_confusao_ajustada <- table(novo_predicoes, test_data$label)

#Printa matriz de confusao para avaliar desempenho do modelo
print(matriz_confusao_ajustada)

