from get_args import Args

def main():
    args_class = Args()
    args = args_class.args
    print("---")
    print(args.data_path)


if __name__ == "__main__":
    main()
    
    import numpy as np
    data = [0,1,2,3,4,5,6,7,8,9]
    data = np.array(data)
    data_split = []
    pred_len = 1
    seq_len = 7
    step_len = 4

    start_index = 0
    end_index = len(data) - seq_len + 1

    for j in range(start_index, end_index, step_len):
        indices = range(j, j+seq_len)
        data_split.append(data[indices])

    print(len(data_split))