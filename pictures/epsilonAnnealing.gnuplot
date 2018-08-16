set terminal epslatex standalone size 12cm,8 cm color colortext font 'Helvetica,10' header "\\usepackage{amsmath} \n   \\usepackage{siunitx} \n \\sisetup{ \n  locale = DE , \n  output-decimal-marker = {.},\n  per-mode = symbol \n }"

set output 'epsilon.tex'

set samples 1000

exploration_initial = 1
exploration_final = 0.1
exploration_evaluation = 0.01
memory_buffer_start_size = 5e4
max_frames = 2e6
annealing_frames = 1e6

set xrange [0:max_frames]
set yrange [0:exploration_initial + 0.05]

set ylabel '$\epsilon$' rotate by 0 offset screen .01
set xlabel '$n_\text{frame}$'

set ytics .2 add (0.1)
set xtics 5e5,5e5 add (5e4, 2.5e5 1)
set mxtics 2
set mytics 2

unset key

set arrow 1 from memory_buffer_start_size, graph 0 to first memory_buffer_start_size, graph 1 ls 0 nohead
set arrow 2 from memory_buffer_start_size + annealing_frames, graph 0 to memory_buffer_start_size + annealing_frames, graph 1 ls 0 nohead

# Slopes and intercepts for epsilon-annealing
m1=-(exploration_initial-exploration_final)/annealing_frames
b1=exploration_initial-m1*memory_buffer_start_size
m2=-(exploration_final-exploration_evaluation)/(max_frames - annealing_frames - memory_buffer_start_size)
b2=exploration_evaluation -m2*max_frames

plot exploration_initial / (x<memory_buffer_start_size) w l lc 1 lw 2,\
     m1*x + b1 / (x>memory_buffer_start_size) / (x <= memory_buffer_start_size + annealing_frames) w l lc 1 lw 2,\
     m2*x + b2 / (x>memory_buffer_start_size + annealing_frames) / (x <= max_frames) w l lc 1 lw 2,\
     0.1 w l ls 0,\
     0.01 w l ls 0


# Run in terminal: latex epsilon.tex && dvips -o epsilon.eps epsilon.dvi && open epsilon.eps

