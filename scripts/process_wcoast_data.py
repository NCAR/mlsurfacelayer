#!/usr/bin/env python
import argparse
#from mlsurfacelayer.data import process_cabauw_data
#from mlsurfacelayer.data import process_idaho_data
from mlsurfacelayer.mvco_data import process_mvco_data
#from mlsurfacelayer.fino_data_2006_2010 import process_fino_2006_2010_data
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input data path.")
    parser.add_argument("-o", "--output", help="Output file.")
    args = parser.parse_args()
    process_mvco_data(args.input, args.output)
    return

if __name__ == "__main__":
    main()
