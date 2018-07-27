set terminal png enhanced size 600,400
set output 'huber.png'

set samples 1000

set xrange [-2.5:2.5]
set yrange [0:*]

set ylabel 'Loss' offset screen .035
set xlabel 'Theta'

set key top left Left

set arrow 1 from 1, graph 0 to first 1, graph 1 ls 0 nohead
set arrow 2 from -1, graph 0 to -1, graph 1 ls 0 nohead

plot x - .5 / (x >= 1) w l lw 2 lc rgb "green" t "Huber loss",\
     -x - .5 / (x <= -1) w l lw 2 lc rgb "green" notitle,\
     .5 * x*x w l lw 2 lc rgb "red" t 'Quadratic loss',\


!open huber.png
