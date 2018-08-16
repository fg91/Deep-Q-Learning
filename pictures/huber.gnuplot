set terminal epslatex standalone size 12cm,8 cm color colortext font 'Helvetica,10' header "\\usepackage{amsmath} \n   \\usepackage{siunitx} \n \\sisetup{ \n  locale = DE , \n  output-decimal-marker = {.},\n  per-mode = symbol \n }"

set output 'huber.tex'

set samples 1000

set xrange [-2.5:2.5]
set yrange [0:*]

set ylabel 'Loss'
set xlabel '$\theta$'

set key top left Left

set xtics 1
set mxtics 2
set ytics 1
set mytics 2

set arrow 1 from 1, graph 0 to first 1, graph 1 ls 0 nohead
set arrow 2 from -1, graph 0 to -1, graph 1 ls 0 nohead

plot x - .5 / (x >= 1) w l lw 2 lc rgb "green" t "Huber loss",\
     -x - .5 / (x <= -1) w l lw 2 lc rgb "green" notitle,\
     .5 * x*x w l lw 2 lc rgb "red" t 'Quadratic loss',\


# Run in terminal: latex huber.tex && dvips -o huber.eps huber.dvi && open huber.eps
