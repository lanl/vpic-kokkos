
import argparse

def setup_args():
    parser = argparse.ArgumentParser()
    # Add a required argument for the TEST timings
    parser.add_argument("--test_file", type = str,
                        help = "Timing data column for TEST advance_p.")
    # Add a required argument for the devel timings
    parser.add_argument("--devel_file", type = str,
                        help = "Timing data column for devel advance_p.")
    # Add a required argument for the output file
    parser.add_argument("--output_file", type = str,
                        help = "Output pathname for the final two-column data per deck.")
    # Add a required argument for deck type
    parser.add_argument("--deck", type = str,
                        help = "Deck type.")

    args = parser.parse_args()
    if args.test_file == None or args.devel_file == None or args.output_file or args.deck == None:
        print("\nArgument error. Execute python3 %s --help for more information.\n" % __file__)
        return False

    return parser

def read_single_file(input_file_name):
    data = []
    input_file = open(input_file_name, "r")
    for line in input_file:
        data.append(line.split()[0])
    input_file.close()
    return data

def write_data_file(args, test_data, devel_data):

    output_file = open(args.output_file, "w")
    output_file.write("# 1: TEST  2: devel\n")
    for row in range(0, len(test_data)):
        if row != len(test_data)-1:
            output_file.write("%s  %s\n" % (test_data[row], devel_data[row]))
        else:
            output_file.write("%s  %s" % (test_data[row], devel_data[row]))

    output_file.close()

def main():

    parser = setup_args()
    if not parser:
        return

    test_data = read_single_file(parser.parse_args().test_file)
    devel_data = read_single_file(parser.parse_args().devel_file)

    if len(test_data) != len(devel_data):
        print("\nERROR: Unfair comparison found. TEST has %d elements and devel has %d.\n", len(test_file), len(devel_data))
        return

    write_data_file(parser.parse_args(), test_data, devel_data)


if __name__ == "__main__":
    main()
