#!/usr/bin/Rscript

print('Running: CORRELATED TOPIC MODEL')

# Packages:
library('tm')
library('ldatuning')
library('topicmodels')
library('magrittr')

# Loading documents
docs <- scan(file = '../../../data/data_schoolofinf/toks/bow2idx.meta', character(), sep = '\n')
# Create corpus
corpus <- Corpus(VectorSource(docs))
# Create a document-term matrix for LDA
dtm <- DocumentTermMatrix(corpus)
inspect(dtm)
rowTotals <- apply(dtm , 1, sum)
dtm.new   <- dtm[rowTotals> 0, ]



# CTM variables
control_list_ctm <- list(
  seed = 5:9,
  nstart = 1,
  best = TRUE,
  verbose=100,
  keep=100,
  iter.max=2000
)


# RUN DIFFERENT NUMBERS OF TOPIC:
print('CTM - 35 ')
ctm <-CTM(k=35, x=dtm.new, control=control_list_ctm)
save(list=c("ctm"), file="./ctm_35.model2.Rdata")
print('Done CTM-35!')
