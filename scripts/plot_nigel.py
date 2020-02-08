import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, realpath
import sys
my_dir = dirname(realpath(__file__))
sys.path.insert(0,my_dir)

import matplotlib_style as mpls

cuda_labels = [ '1 V100', 
            '2 V100s', 
            '4 V100s' ]
power9_labels = ['Original', 'Vectorized', 'Kokkos']
epyc_labels = ['Original', 'Vectorized', 'Kokkos']
thunderX2_labels = ['Original', 'Kokkos']
knl_labels = ['Original', 'Vectorized', 'Kokkos']

cuda = [99.4822, 54.5741, 28.5621] # Tesla V100s ( 1, 2, 4 ) GPUs
power9 = [131.423, 73.1652, 182.02] # Power9 40 cores
epyc = [93.4852, 56.621, 115.926] 
thunderX2 = [127.043, 162.902]
knl = [239.296, 138.075, 344.746]
sklake = [157.701, 61.43, 179.096]
skx_plat = [106.769, 54.416, 128.605]
rome = [54.5867, 39.8546, 63.6075]


fig, ax = mpls.init_plot(style='poster',dpi=150, frac=(1.5,0.5))

x = np.arange(3+3+2+3+3)

ax.bar(x[0:3], knl, label='Knights Landing', color='#0071C5')
ax.bar(x[3:6], cuda, label='Tesla V100s', color='#76B900')
ax.bar(x[6:9], epyc, label='EPYC', color='#990000')
ax.bar(x[9:12], power9, label='Power9', color='#051243')
ax.bar(x[12:14], thunderX2, label='Cavium Thunder-X2', color='yellow')

plt.xticks(np.arange(min(x),max(x)+1,1.0))

labels = [item.get_text() for item in ax.get_xticklabels()]
labels = knl_labels
for i in range(0,len(cuda)):
    labels.append(cuda_labels[i])
for i in range(0,len(epyc)):
    labels.append(epyc_labels[i])
for i in range(0,len(power9)):
    labels.append(power9_labels[i])
for i in range(0,len(thunderX2)):
    labels.append(thunderX2_labels[i])
#labels.append(labels2)

ax.set_xticklabels(labels,rotation='vertical')
ax.set_ylabel('Time (s)')
ax.set_title('~66 million particles, 500 time steps')
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha='center', va='bottom')

mpls.auto_layout(fig,ax,ywinbuffer=[0.05,0.05])


plt.savefig('result_plot', dpi=300)
plt.show()
