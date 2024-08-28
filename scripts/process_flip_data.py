#!/usr/bin/env python
import argparse
from mlsurfacelayer.flip_data import process_flip_data
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input data path.")
    parser.add_argument("-o", "--output", help="Output file.")
    args = parser.parse_args()
    process_flip_data(args.input, args.output)
    return

if __name__ == "__main__":
    main()
