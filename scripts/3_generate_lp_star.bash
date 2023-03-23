FOLDER=DATA/LP_STAR

mkdir -p $FOLDER
for number in 0 1 2 3 
do
   python sample/sample.py --vocab_file sample/vocab.txt --output_file $FOLDER/prop_examples_$number.txt --min_pred_num 5 --max_pred_num 30 --algo LP_STAR --example_num 100 --balance_by_depth --max_depth 6 & 
done
wait
python dataset.py --file_name $FOLDER/prop_examples_INDEX.txt --file_range 40 --max_depth_during_train 6 --final_file_name $FOLDER/prop_examples.balanced_by_backward.max_6.json --control_num 100 --depth depth