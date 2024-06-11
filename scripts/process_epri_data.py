#!/usr/bin/env python
import argparse
from mlsurfacelayer.epri_data import  process_epri_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input data path.")
    parser.add_argument("-s", "--site", default="epri", help="Site type: epri")
    parser.add_argument("-o", "--output", help="Output file.")
    args = parser.parse_args()
    if args.site == "epri":
        process_epri_data(args.input, args.output)
    return

if __name__ == "__main__":
    main()
