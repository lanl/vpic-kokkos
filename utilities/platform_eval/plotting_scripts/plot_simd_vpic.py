import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

plt.style.use('tableau-colorblind10')

parser = argparse.ArgumentParser(description="VPIC runtime image generator")
parser.add_argument('profiles', metavar='N', type=str, nargs='+', help='VPIC output csv')
parser.add_argument('--output-fname', type=str, nargs=1, default='vpic_profiles.eps', help='Filename for writing profile output')
args = parser.parse_args()

df = pd.read_csv(args.profiles[0])
print(df.columns)

#vec_map = {'neon': 'vec', 'avx2': 'simd_avx2'}
#df = df.replace({'Vectorization': vec_map})
chip_map = {'sprddr': 'SPR\nDDR', 'sprhbm': 'SPR\nHBM', 'a64fx': 'A64FX', 'ampere1': 'Ampere\nAltra', 'epyc7742': 'EPYC\nRome', 'epyc7763': 'EPYC\nMilan', 'icelake': 'Ice\nLake', 'thunderx2': 'ThunderX2', 'v100': 'V100', 'a100': 'A100', 'h100': 'H100', 'mi100': 'MI100', 'mi250': 'MI250', 'grace': "Grace", 'mi300a': 'MI300A'}
chip_id_map = {'sprddr': 1, 'sprhbm': 2, 'a64fx': 6, 'ampere1': 7, 'epyc7742': 3, 'epyc7763': 4, 'icelake': 0, 'thunderx2': 5, 'v100': 6, 'a100': 7, 'h100': 8, 'mi100': 9, 'mi250': 10, 'grace': 11, 'mi300a': 12}
df['Chip ID'] = [chip_id_map[x] for x in df['Chip']]
df = df.replace({'Chip': chip_map})
vec_map = {'auto': 'Auto', 'guided': 'Guided', 'manual': 'Manual', 'ad-hoc': 'Ad Hoc'}
df = df.replace({'Vectorization': vec_map})
sort_map = {'random': 'Random', 'standard': 'Standard', 'strided': 'Strided', 'tiled-strided': 'Tiled Strided'}
df = df.replace({'Sort Order': sort_map})

df = df.sort_values(by=['Chip ID'])

df = df[df['Step'] == 1001]
df = df[df['Timer'] == 'advance_p']
#print(df[['Chip', 'Time Since Last Restore', 'Vectorization']].to_string())

fig, (axtop, axbot) = plt.subplots(ncols=1, nrows=2, sharex=True, height_ratios=[1,3])
bar_plot = sns.barplot(ax=axtop, data=df, x='Chip', y='Time Since Last Restore', hue='Vectorization', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual', 'Ad Hoc'])
bar_plot = sns.barplot(ax=axbot, data=df, x='Chip', y='Time Since Last Restore', hue='Vectorization', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual', 'Ad Hoc'])
axtop.set_ylim(150, 1000)
axbot.set_ylim(0, 150)
axtop.get_xaxis().set_visible(False)
axbot.get_legend().remove()
axtop.legend(ncols=2, fontsize=12)
axtop.set_ylabel(None)
axbot.set_ylabel(None)
axbot.set_xlabel(None)
#fig.text(0.02, 0.55, "Particle Push Runtime (s)", va="center", rotation="vertical")
axtop.xaxis.tick_top()
axbot.xaxis.tick_bottom()
axbot.set_xticklabels(axbot.get_xticklabels(), size=12)
axtop.set_yticklabels(axtop.get_yticklabels(), size=12)
axbot.set_yticklabels(axbot.get_yticklabels(), size=12)
plt.xlabel(None)
fig.supylabel("Particle Push Runtime (s)", fontsize=12)

plt.tight_layout()
plt.show()
#fig.savefig(args.output_fname[0])
