#!/bin/bash

years=(2015 2016)

# List of DA types
BERT_da_types=("BERT-nouns" "BERT-adverbs" "BERT-nouns_adverbs" "BERT-aspect" "BERT-aspect_adverbs")
CBERT_da_types=("CBERT-nouns" "CBERT-adverbs" "CBERT-nouns_adverbs" "CBERT-aspect" "CBERT-aspect_adverbs")
BERTexpand_da_types=("BERT_expand-nouns" "BERT_expand-adverbs" "BERT_expand-nouns_adverbs" "BERT_expand-aspect" "BERT_expand-aspect_adverbs")


base_command="python remaining_idx.py"

# Loop through each year and each DA type to run the command
for year in "${years[@]}"
do
  for da_type in "${[@]}"
  do
    command="$base_command --year $year --da_type $da_type"
    echo "Running command: $command"
    $command
  done
done