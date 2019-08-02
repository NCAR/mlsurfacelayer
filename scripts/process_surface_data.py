import argparse
from mlsurfacelayer.data import process_cabauw_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input data path.")
    parser.add_argument("-s", "--site", default="cabauw", help="Site type: cabauw, nrel, idaho")
    parser.add_argument("-o", "--output", help="Output file.")
    parser.add_argument("-r", "--refl", action="store_true", help="Change sign of Counter Gradient Fluxes")
    args = parser.parse_args()
    if args.site == "cabauw":
        process_cabauw_data(args.input, args.output, reflect_counter_gradient=args.refl)

    return

if __name__ == "__main__":
    main()