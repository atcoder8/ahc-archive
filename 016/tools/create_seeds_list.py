import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This program creates a list of seeds.")
    parser.add_argument("seed_num", help="Number of seeds.", type=int)
    parser.add_argument(
        "--start", "-s", type=int, default=0, help="Starting number of the seed list."
    )

    args = parser.parse_args()
    seed_num: int = args.seed_num
    start: int = args.start

    for i in range(start, start + seed_num):
        print(i)
