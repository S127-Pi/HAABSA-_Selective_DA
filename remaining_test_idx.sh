#!/bin/bash

years=(2015 2016)

# List of DA types
BERT_da_types=("BERT-random" "BERT-nouns" "BERT-adverbs" "BERT-nouns_adverbs" "BERT-aspect_adverbs")
CBERT_da_types=("CBERT-random" "CBERT-nouns" "CBERT-adverbs" "CBERT-nouns_adverbs" "CBERT-aspect_adverbs")
BERTexpand_da_types=("BERT_expand-random" "BERT_expand-nouns" "BERT_expand-adverbs" "BERT_expand-nouns_adverbs" "BERT_expand-aspect_adverbs")
all_da_types=("${BERT_da_types[@]}" "${CBERT_da_types[@]}" "${BERTexpand_da_types[@]}")

base_command="python remaining_idx.py"
# Loop through each year and each DA type to run the command
for year in "${years[@]}"
do
  for da_type in "${all_da_types[@]}"
  do
    command="$base_command --year $year --da_type $da_type"
    echo "Running command: $command"
    $command
  done
done