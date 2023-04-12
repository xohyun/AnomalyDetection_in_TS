for eee in 20 30 50 100 200 500
do
    for seq in 15 20 30 45 50 60
    do
        for step in 1 2 3 4 5 6
        do
            python main.py --csv_path="/content/drive/MyDrive/score/hold/" --model=Boosting_aug --epoch=$eee --dataset=NAB --choice_data='hold' --seq_len=$seq --step_len=$step --score='var_corr'
        done
    done
done