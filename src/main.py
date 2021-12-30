import sys

sys.path.append('../library/')  # Add custom library to PYTHONPATH.

from train_model import train


def main():
    train()


if __name__ == "__main__":
    main()
