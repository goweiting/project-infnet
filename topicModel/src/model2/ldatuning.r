#!/usr/bin/Rscript

# Packages:
library('tm')
library('ldatuning')
library('topicmodels')
#library('magrittr')

# Loading documents
docs <- scan(file = '../../../data/data_schoolofinf/toks/bow2idx.meta', character(), sep = '\n')
# Create corpus
corpus <- Corpus(VectorSource(docs))
# Create a document-term matrix for LDA
dtm <- DocumentTermMatrix(corpus)
inspect(dtm)
rowTotals <- apply(dtm , 1, sum)
dtm.new   <- dtm[rowTotals> 0, ]


# Metrics for LDA
control_list_gibbs <- list(
  burnin = 2500,
  iter = 5000,
  seed = 0:4,
  nstart = 5,
  best = TRUE
)


# Executing larger number of topics: NOTE: TOPIC_NUMBER2
system.time(
  topic_number1 <- FindTopicsNumber(
    dtm.new,
    topics = c(seq(5,10,1), seq(12, 20, 2), seq(25, 50, 5), seq(60,100, 10)),
    metrics = c( "Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
    method = "Gibbs",
    control = control_list_gibbs,
    mc.cores = 10L,
    verbose = TRUE
  )
)
save(list=c("topic_number1"), file="./tuning.topic_number.rdata")
print("===COMPLETED EXECUTION===")
