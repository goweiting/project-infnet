#!/usr/bin/Rscript

print('Running CTM tuning on fullpub:')

# Packages:
library(tm)
library(ldatuning)
library(topicmodels)
library(parallel)

# Loading documents
docs <- scan(file = '../data/fullpub/combined_toks.txt', character(), sep = '\n')
# Create corpus
corpus <- Corpus(VectorSource(docs))
# Create a document-term matrix for LDA
dtm <- DocumentTermMatrix(corpus)


# CTM variables
control_list_ctm <- list(
  seed = 5:9,
  nstart = 5,
  best = TRUE
)


# RUN DIFFERENT NUMBERS OF TOPIC:
# print('CTM - 10 ')
# ctm_20 <-CTM(k=10, x=dtm, control=control_list_ctm)
# save(list=c("ctm_10"), file="./ctm.10.Rdata")
# print('Done CTM-10!')

# print('CTM - 20 ')
# ctm_20 <-CTM(k=20, x=dtm, control=control_list_ctm)
# save(list=c("ctm_20"), file="./ctm.20.Rdata")
# print('Done CTM-20!')


ks <- c(seq(from=30,to=100,by=10))

run.worker <- function(x){
    # Takes in a value of k (number of topics) and start the 
    # computation for CTM
    output <- CTM(k=x, x=dtm, control=control_list_ctm)
    save(output, file=sprintf('ctm.%d',x))
    print(sprintf('ctm.%d',x))
}

print("STARTING JOB")
print(ks)
mclapply(ks, run.worker)
