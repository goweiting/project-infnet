#!/usr/bin/Rscript

print('Running LDA tuning on fullpub:')

# Packages:
library('tm')
library('ldatuning')
library('topicmodels')
library('magrittr')

# Loading documents
docs <- scan(file = '../data/fullpub/combined_toks.txt', character(), sep = '\n')
# Create corpus
corpus <- Corpus(VectorSource(docs))
# Create a document-term matrix for LDA
dtm <- DocumentTermMatrix(corpus)

# Metrics for LDA
control_list_gibbs <- list(
  burnin = 2500,
  iter = 5000,
  seed = 0:4,
  nstart = 5,
  best = TRUE
)

# Searching for the number of topics
# system.time(
#   topic_number <- FindTopicsNumber(
#     dtm,
#     topics = c(seq(5,10,1), seq(12, 20, 2), seq(25, 50, 5)),
#     metrics = c( "Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
#     method = "Gibbs",
#     control = control_list_gibbs,
#     mc.cores = 40L,
#     verbose = TRUE
#   )
# )

# Save the model for future use!
# save(list=c("topic_number"), file="./tuning.topic_number.rdata")


# Executing larger number of topics: NOTE: TOPIC_NUMBER2
system.time(
  topic_number2 <- FindTopicsNumber(
    dtm,
    topics = c(seq(60,100, 10)),
    metrics = c( "Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
    method = "Gibbs",
    control = control_list_gibbs,
    mc.cores = 40L,
    verbose = TRUE
  )
)
print("===COMPLETED EXECUTION===")