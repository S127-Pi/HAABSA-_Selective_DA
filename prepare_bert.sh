#!/bin/bash

years=(2015 2016)

# List of DA types
BERT_da_types=("BERT-nouns" "BERT-adverbs" "BERT-nouns_adverbs" "BERT-aspect" "BERT-aspect_adverbs")
CBERT_da_types=("CBERT-nouns" "CBERT-adverbs" "CBERT-nouns_adverbs" "CBERT-aspect" "CBERT-aspect_adverbs")
BERTprepend_da_types=("BERTprepend-nouns" "BERTprepend-adverbs" "BERTprepend-nouns_adverbs" "BERTprepend-aspect" "BERTprependaspect_adverbs")
BERTexpand_da_types=("BERTexpand-nouns" "BERTexpand-adverbs" "BERTexpand-nouns_adverbs" "BERTexpand-aspect" "BERTexpand-aspect_adverbs")

base_command="python prepare_bert.py"

# Loop through each year and each DA type to run the command
for year in "${years[@]}"
do
  for da_type in "${BERT_da_types[@]}"
  do
    command="$base_command --year $year --da_type $da_type"
    echo "Running command: $command"
    $command
  done
done