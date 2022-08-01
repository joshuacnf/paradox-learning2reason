# this takes 10X time than 1_generate_rp.bash

FOLDER=DATA/RP_10X

mkdir -p $FOLDER
for number in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
do
   python sample/sample.py --vocab_file sample/vocab.txt --output_file $FOLDER/prop_examples_$number.txt --min_pred_num 5 --max_pred_num 30 --algo RP --example_num 10000 --balance_by_depth --max_depth 6 & 
done
wait


FOLDER=DATA/RP_10X
for number in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
do
   python sample/balance.py --input_file $FOLDER/prop_examples_$number.txt  --output_file $FOLDER/prop_examples_$number.txt.balanced_rulenum & 
done
wait


python sample/downsample.py