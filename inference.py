# inference.py
# training cnn model and evaluate to test dataset and return result.

import argparse
import warnings

from tqdm import tqdm

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="excute inference.py")
    parser.add_argument('--model_type', type=str, default=None, help='select model type [not yet]')
    parser.add_argument('is_valid', type=str2bool, help='select dataset [True=valid(splited by train.csv), False=test.csv]')

    args = parser.parse.args()
    model_type = args.model_type
    is_valid = args.is_valid