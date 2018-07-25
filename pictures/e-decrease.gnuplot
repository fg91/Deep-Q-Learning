set terminal png enhanced size 600,400
set output 'epsilon.png'

set lmargin at screen .1
set rmargin at screen .9

set samples 1000

set xrange [0:2e6]
set yrange [0:1.05]

set ylabel 'epsilon' offset screen .035
set xlabel 'frameNumber'

set ytics .2 add (0.1)

unset key

# Parameter for epsilon-decrease
m1=-0.9/1e6
b1=1-m1*5000
m2=-(0.1-0.01)/(2e6 - 1e6 - 5000)
b2=0.1 - m2*(1e6 + 5000)

plot 1 / (x<5000) w l lc 2 lw 2,\
     m1*x + b1 / (x>5000) / (x <= 1e6) w l lc 2 lw 2,\
     m2*x + b2 / (x>1e6) / (x <= 2e6) w l lc 2 lw 2,\
     0.1 w l ls 0,\
     0.01 w l ls 0

!open epsilon.png
