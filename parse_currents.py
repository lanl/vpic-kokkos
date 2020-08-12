import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file_1", type=str, help="First grepped column input.")

particle_col = 1
current_sum_col = 7

def read_input(file_1):

    input_1 = open(file_1, "r")

    all_lines_1 = input_1.readlines()

    data_values = []
    previous_index = None
    for row in range(0, len(all_lines_1), 3):
        xyz = all_lines_1[row].split()
        yzx = all_lines_1[row+1].split()
        zxy = all_lines_1[row+2].split()
        if int(xyz[particle_col]) == previous_index:
            data_values[-1][1] += float(xyz[current_sum_col])
            data_values[-1][2] += float(yzx[current_sum_col])
            data_values[-1][3] += float(zxy[current_sum_col])
        else:
            data_values.append([xyz[particle_col], float(xyz[current_sum_col]), float(yzx[current_sum_col]), float(zxy[current_sum_col])])

        previous_index = int(data_values[-1][0])

    input_1.close()
    print("\nCurrents read. Now writing data.\n")

    output_1 = open(file_1, "w")
    for lst in data_values:
        output_1.write("Particle %s: jx, jy, jz = %e, %e, %e\n" % (lst[0], lst[1], lst[2], lst[3]) )
    output_1.close()

    print("\nData written.\n")


def main():

    args = parser.parse_args()

    data_values = read_input(args.file_1)


if __name__ == "__main__":
    main()
