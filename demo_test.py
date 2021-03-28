import os

from relation_extraction.test import test
from relation_extraction.hparams import hparams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def main():
    test(hparams)


if __name__ == '__main__':
    main()
