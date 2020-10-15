#!/usr/bin/env python
import argparse
from mlsurfacelayer.data import process_cabauw_data
from mlsurfacelayer.data import process_idaho_data
from mlsurfacelayer.fino_data import process_fino_data
from mlsurfacelayer.fino_data_2006_2010 import process_fino_2006_2010_data
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input data path.")
    parser.add_argument("-s", "--site", default="cabauw", help="Site type: cabauw, idaho, fino")
    parser.add_argument("-o", "--output", help="Output file.")
    parser.add_argument("-r", "--refl", action="store_true", help="Change sign of Counter Gradient Fluxes")
    parser.add_argument("-w", "--wind", default="30Min", help="Time-averaging window in pandas time units. 30Min is the default.")
    args = parser.parse_args()
    if args.site == "cabauw":
        process_cabauw_data(args.input, args.output, reflect_counter_gradient=args.refl, average_period=args.wind)
    if args.site == "idaho":
        process_idaho_data(args.input, args.output, average_period=args.wind)
    if args.site == "fino":
        process_fino_data(args.input, args.output, reflect_counter_gradient=args.refl)
    if args.site == "fino_2006_2010":
       process_fino_2006_2010_data(args.input, args.output, reflect_counter_gradient=args.refl)
    return

if __name__ == "__main__":
    main()
