# import argparse

# def parse_args():
#     args = argparse.ArgumentParser()
#     args.add_argument("--paths", nargs="+", help="Desc")
#     args.add_argument("--seg-iter", type=int, required=False, help="Desc")
#     args.add_argument("--bool", type=bool, default=False, help="Desc")
#     args.add_argument(
#         "--frac",
#         metavar="float",
#         type=lambda x: float(x) > 0 and float(x) <= 1.0,
#         help="Between zero and one",
#     )
#     return args.parse_known_args()

# def main():
#     args, unknown_args = parse_args()


# if __name__ == "__main__":
#     main()
