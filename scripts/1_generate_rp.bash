FOLDER=DATA/RP

mkdir -p $FOLDER
for number in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 # sample with 40 processes
do
   python sample/sample.py --vocab_file sample/vocab.txt --output_file $FOLDER/prop_examples_$number.txt --min_pred_num 5 --max_pred_num 30 --algo RP --example_num 1000 --balance_by_depth --max_depth 6 & 
done
wait
python dataset.py --file_name $FOLDER/prop_examples_INDEX.txt --file_range 40 --max_depth_during_train 6 --final_file_name $FOLDER/prop_examples.balanced_by_backward.max_6.json --control_num 1000 --depth backward_depth

# python sample/sample.py --vocab_file sample/vocab.txt --output_file logic_bert/test_examples.txt --min_pred_num 5 --max_pred_num 30 --algo RP --example_num 100 --balance_by_depth --max_depth 10