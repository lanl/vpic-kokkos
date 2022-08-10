#!/usr/bin/perl
use strict;
use warnings;

my $slurm_file = $ARGV[0];
my $plot_data   = "vpic_plot_data.dat";
my $plot_script = "vpic_plot_script.p";
my $vpic_plot   = "vpic_perf_plot.png";
my $extract_data_script = "\$HOME/VPIC/vpic-kokkos/TestScripts/extract_output_data.pl"; 

system("perl $extract_data_script $slurm_file > $plot_data"); 

open(my $out, ">", $plot_script) or die "can't open $plot_script"; 

print $out "set term png\n";
print $out "set output \"$vpic_plot\"\n";
print $out "set multiplot layout 2,2 rowsfirst title \"{/:Bold=15 VPIC 2.0 Weak Scaling Benchmarks}\" \n";
print $out "set key box opaque \n";
print $out "set key left box\n";

print $out "set xlabel \"\#MPI Ranks\" offset 0,0 \n";
print $out "set ylabel \"Run Time, sec.\"   offset 0,0 \n";
print $out "set title \"Total Run Time\" font \",12\"\n";
print $out "plot \"$plot_data\" u 2:4 w lp title \"Chicoma\" lt 1 \n";

print $out "unset title \n";

print $out "set title \"Advance Particles\" font \",12\"\n";
print $out "plot \"$plot_data\" u 2:5 w lp title \"Chicoma\" lt 2 \n";

print $out "unset title \n";

print $out "set title \"Advance Boundary Particles\" font \",12\"\n";
print $out "plot \"$plot_data\" u 2:6 w lp title \"Chicoma\" lt 3 \n";

print $out "unset title \n";

print $out "set title \"Sort Particles\" font \",12\"\n";
print $out "plot \"$plot_data\" u 2:7 w lp title \"Chicoma\" lt 4 \n";

print $out "unset multiplot \n";

close $out or die "$out: $!";

system("gnuplot	$plot_script");
