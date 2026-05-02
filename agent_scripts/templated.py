import argparse

DEFAULT_NAME_KEY = "name"

"""Example generated script."""

def run(params):
    print(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    params = {DEFAULT_NAME_KEY: args.name}
    run(params)
