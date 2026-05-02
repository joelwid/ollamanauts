import argparse

"""Example generated script."""

def run(params):
    print(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    params = {"name": args.name}
    run(params)
