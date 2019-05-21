set term post eps enhanced 22 color
set output 'Weibel_e-1d-comp.eps'

set xlabel '# steps'
set ylabel 'W_B'
set title "Weibel instability (2 mpi)"
plot 'energies-gpu-2mpi' u 1:($5+$6+$7) w l lw 3 t 'gpu','energies-cpu-2mpi' u 1:($5+$6+$7) w l lw 3 t 'cpu'