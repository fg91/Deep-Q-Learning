set terminal png enhanced size 600,400
set output 'epsilon.png'

set samples 1000

explorationInitial = 1
explorationFinal = 0.1
explorationInitialnference = 0.01
memoryBufferStartSize = 5e4
maxFrames = 2e6
AnnealingFrames = 1e6

set xrange [0:maxFrames]
set yrange [0:explorationInitial + 0.05]

set ylabel 'epsilon' offset screen .035
set xlabel 'frameNumber'

set ytics .2 add (0.1)
set xtics 5e5,5e5 add (5e4)

unset key

set arrow 1 from memoryBufferStartSize, graph 0 to first memoryBufferStartSize, graph 1 ls 0 nohead
set arrow 2 from memoryBufferStartSize + AnnealingFrames, graph 0 to memoryBufferStartSize + AnnealingFrames, graph 1 ls 0 nohead

# Slopes and intercepts for epsilon-annealing
m1=-(explorationInitial-explorationFinal)/AnnealingFrames
b1=explorationInitial-m1*memoryBufferStartSize
m2=-(explorationFinal-explorationInitialnference)/(maxFrames - AnnealingFrames - memoryBufferStartSize)
b2=explorationInitialnference -m2*maxFrames

plot explorationInitial / (x<memoryBufferStartSize) w l lc 2 lw 2,\
     m1*x + b1 / (x>memoryBufferStartSize) / (x <= memoryBufferStartSize + AnnealingFrames) w l lc 2 lw 2,\
     m2*x + b2 / (x>memoryBufferStartSize + AnnealingFrames) / (x <= maxFrames) w l lc 2 lw 2,\
     0.1 w l ls 0,\
     0.01 w l ls 0


!open epsilon.png
