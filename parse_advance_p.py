
import argparse as argp

def setup_args():

    bad_args = False
    parser = argp.ArgumentParser()
    # Add required argument for the file location
    parser.add_argument("-APF" , "--advance_p_file", type = str,
                        help = "Grepped stderr output filepath for advance_p timings.")
    # Add required argument for the file location
    parser.add_argument("-OF" , "--output_file", type = str,
                        help = "Output file to write timing data column.")
    # Add required argument for the deck
    parser.add_argument("-D", "--deck", type = str,
                        help = "Deck type.")
    # Add required argument for the version
    parser.add_argument("-V", "--version", type = str,
                        help = "Version/branch type.")

    if parser.parse_args().advance_p_file == None:
        bad_args = True
    if parser.parse_args().advance_p_file == None:
        bad_args = True
    if parser.parse_args().deck == None:
        bad_args = True
    if parser.parse_args().version == None:
        bad_args = True

    if bad_args:
        print("\nError running script. Execute python3 %s --help for details.\n" % __file__)
        return False

    return parser

def read_input_file(args):

    # Take the Per column (ordered from 1)
    column = 6
    input_file = open(args.advance_p_file, "r")
    timing_data = []

    for line in input_file:
        timing_data.append(line.split()[column-1])

    if len(timing_data) == 0:
        print("\nWarning: No data in file.\n")

    input_file.close()
    return timing_data

def write_output_data(args, data):

    output_file = open(args.output_file, "w")
    for d in range(0, len(data)):
        if d != len(data)-1 :
            output_file.write("%s\n" % data[d])
        else:
            output_file.write("%s" % data[d])

def main():

    parser = setup_args()
    if not parser:
        return


    timing_data = read_input_file(parser.parse_args())

    if len(timing_data) == 0:
        return

    write_output_data(parser.parse_args(), timing_data)


if __name__ == "__main__":
    main()
