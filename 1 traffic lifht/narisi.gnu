set style line 12 lc rgb '#808080' lt 0 lw 1
set term tikz standalone

set grid back ls 12
set key right top
set encoding utf8
set style fill transparent solid 0.5 border 
set ylabel "$\tilde{v}$"
set xlabel "$\tilde{t}$"
plot [x=0:1] (-3/2)*(1-0.0)*x**2 +3*(1-0.0)*x + 0 t "$\\tilde{v_{1}}=0$" w l
set out 'v_enak1.pdf'

