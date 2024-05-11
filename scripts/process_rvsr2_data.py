#!/usr/bin/env python
import argparse
#from mlsurfacelayer.data import process_cabauw_data
#from mlsurfacelayer.data import process_idaho_data
from mlsurfacelayer.rvsr2_data import process_rvsr2_data
#from mlsurfacelayer.fino_data_2006_2010 import process_fino_2006_2010_data

# -i /Volumes/SuesRoo/mlsurfacelayer/wcoastData/raw/RVSR_20min_CASPER17_West.csv
# -o /Volumes/SuesRoo/mlsurfacelayer/wcoastData/csv/rvrs_casper.csv
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input data path.")
    parser.add_argument("-o", "--output", help="Output file.")
    args = parser.parse_args()
    process_rvsr2_data(args.input, args.output)
    return

if __name__ == "__main__":
    main()
