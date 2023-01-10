from get_args import Args

def main():
    args_class = Args()
    args = args_class.args
    print("---")
    print(args.data_path)


if __name__ == "__main__":
    main()