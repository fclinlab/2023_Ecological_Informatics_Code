import argparse


parser = argparse.ArgumentParser()
parser.add_argument("band_set", help="這是第 1 個引數，請輸入執行dataset")
parser.add_argument("band_num", help="這是第 2 個引數，請輸入整數", type = int)
parser.add_argument("batch_size", help="這是第 3 個引數，請輸入整數", type = int)
parser.add_argument("standard_param", help="這是第 4 個引數，請輸入整數", type = int)
args = parser.parse_args()