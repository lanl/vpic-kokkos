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
vec_map = {'auto': 'Auto', 'guided': 'Guided', 'manual': 'Manual', 'ad-hoc': 'Ad-Hoc'}
df = df.replace({'Mode': vec_map})
df['Run'] = np.arange(len(df)) % 50 + 1
df['Improvement'] = np.repeat(0, len(df))
for i in range(len(df)):
  if df.at[i, 'Mode'] == 'Auto':
    df.at[i, 'Improvement'] = 1.0
  elif df.at[i, 'Mode'] == 'Guided':
    df.at[i, 'Improvement'] = df.at[i, 'Kernel time'] / df.at[i-50, 'Kernel time']
  elif df.at[i, 'Mode'] == 'Manual':
    df.at[i, 'Improvement'] = df.at[i, 'Kernel time'] / df.at[i-100, 'Kernel time']

print(df.columns)
print(df)
print(df[df['Kernel'] == 'PI_REDUCE'].groupby(['Chip', 'Mode'])['Improvement'].mean())
df = df.sort_values(by=['Chip ID'])

print(df[['Chip', 'Kernel', 'Kernel time', 'Improvement']])

imp_fig, (ax0, ax1, ax2) = plt.subplots(ncols=1, nrows=3, sharex=True)
axpy_plot = sns.barplot(ax=ax0, data=df[df['Kernel'] == 'AXPY'], x='Chip', y='Improvement', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'])
planck_plot = sns.barplot(ax=ax1, data=df[df['Kernel'] == 'PLANCKIAN'], x='Chip', y='Improvement', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'], legend=False)
fir_plot = sns.barplot(ax=ax2, data=df[df['Kernel'] == 'PI_REDUCE'], x='Chip', y='Improvement', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'], legend=False)
ax0.set_ylabel(None)
ax1.set_ylabel("Normalized Runtime", fontsize=14)
ax2.set_ylabel(None)
ax0.set_title("AXPY",      fontsize=12, loc='right', y=0.75)
ax1.set_title("PLANCKIAN",       fontsize=12, loc='right', y=0.75)
ax2.set_title("PI_REDUCE", fontsize=12, loc='right', y=0.75)
ax2.set_xlabel(None)
ax0.legend(ncol=3, loc='upper center')
plt.tight_layout()

axpy_fig = plt.figure()
axpy_plot = sns.barplot(data=df[df['Kernel'] == 'AXPY'], x='Chip', y='Improvement', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'])
plt.ylabel("Normalized Runtime", fontsize=14)
plt.xlabel(None)
ax = axpy_fig.get_axes()[0]
ax.set_yticklabels(ax.get_yticklabels(), size=12)
ax.set_xticklabels(ax.get_xticklabels(), size=12)
plt.legend(fontsize=12)
plt.tight_layout()

planck_fig = plt.figure()
planck_plot = sns.barplot(data=df[df['Kernel'] == 'PLANCKIAN'], x='Chip', y='Improvement', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'])
plt.ylabel("Normalized Runtime", fontsize=14)
plt.xlabel(None)
ax = planck_fig.get_axes()[0]
ax.set_yticklabels(ax.get_yticklabels(), size=12)
ax.set_xticklabels(ax.get_xticklabels(), size=12)
plt.legend(fontsize=12)
plt.tight_layout()

pi_reduce_fig = plt.figure()
pi_reduce_plot = sns.barplot(data=df[df['Kernel'] == 'PI_REDUCE'], x='Chip', y='Improvement', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'])
plt.ylabel("Normalized Runtime", fontsize=14)
plt.xlabel(None)
ax = pi_reduce_fig.get_axes()[0]
ax.set_yticklabels(ax.get_yticklabels(), size=12)
ax.set_xticklabels(ax.get_xticklabels(), size=12)
plt.legend(fontsize=12)
plt.tight_layout()


fig, (ax0, ax1, ax2) = plt.subplots(ncols=1, nrows=3, sharex=True)
axpy_plot = sns.barplot(ax=ax0, data=df[df['Kernel'] == 'AXPY'], x='Chip', y='Kernel time', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'], legend=False)
planck_plot = sns.barplot(ax=ax1, data=df[df['Kernel'] == 'FIR'], x='Chip', y='Kernel time', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'])
fir_plot = sns.barplot(ax=ax2, data=df[df['Kernel'] == 'PLANCKIAN'], x='Chip', y='Kernel time', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'], legend=False)
ax0.set_ylabel(None)
ax1.set_ylabel("Runtime (s)", fontsize=14)
ax2.set_ylabel(None)
ax0.set_title("AXPY",      fontsize=12, loc='right', y=0.75)
ax1.set_title("FIR",       fontsize=12, loc='right', y=0.75)
ax2.set_title("PLANCKIAN", fontsize=12, loc='right', y=0.75)
ax2.set_xlabel(None)
ax1.legend(ncol=3, loc='upper center')
plt.tight_layout()

#imp_fig2, (ax0, ax1, ax2) = plt.subplots(ncols=1, nrows=3, sharex=True)
#axpy_plot = sns.barplot(ax=ax0, data=df[df['Kernel'] == 'PRESSURE'], x='Chip', y='Improvement', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'], legend=False)
#planck_plot = sns.barplot(ax=ax1, data=df[df['Kernel'] == 'IF_QUAD'], x='Chip', y='Improvement', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'], legend=False)
#fir_plot = sns.barplot(ax=ax2, data=df[df['Kernel'] == 'PI_REDUCE'], x='Chip', y='Improvement', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'])
#ax0.set_ylabel(None)
#ax1.set_ylabel("Runtime (s)", fontsize=14)
#ax2.set_ylabel(None)
#ax0.set_title("PRESSURE",      fontsize=12, loc='right', y=0.75)
#ax1.set_title("IF_QUAD",       fontsize=12, loc='right', y=0.75)
#ax2.set_title("PI_REDUCE", fontsize=12, loc='right', y=0.75)
#ax2.set_xlabel(None)
#ax2.legend(ncol=3, loc='upper center')

fig2, (ax0, ax1, ax2) = plt.subplots(ncols=1, nrows=3, sharex=True)
axpy_plot = sns.barplot(ax=ax0, data=df[df['Kernel'] == 'PRESSURE'], x='Chip', y='Kernel time', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'], legend=False)
planck_plot = sns.barplot(ax=ax1, data=df[df['Kernel'] == 'IF_QUAD'], x='Chip', y='Kernel time', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'], legend=False)
fir_plot = sns.barplot(ax=ax2, data=df[df['Kernel'] == 'PI_REDUCE'], x='Chip', y='Kernel time', hue='Mode', order=['A64FX', 'EPYC\nMilan', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A'], hue_order=['Auto', 'Guided', 'Manual'])
ax0.set_ylabel(None)
ax1.set_ylabel("Runtime (s)", fontsize=14)
ax2.set_ylabel(None)
ax0.set_title("PRESSURE",      fontsize=12, loc='right', y=0.75)
ax1.set_title("IF_QUAD",       fontsize=12, loc='right', y=0.75)
ax2.set_title("PI_REDUCE", fontsize=12, loc='right', y=0.75)
ax2.set_xlabel(None)
ax2.legend(ncol=3, loc='upper center')

#sortfig = plt.figure()
#sort_plot = sns.barplot(data=df, x='Chip', y='Time Since Last Restore', hue='Sort Order', order=['A100', 'H100', 'MI100', 'MI250', 'MI300A'], hue_order=['Random', 'Standard', 'Strided', 'Tiled Strided'])
#plt.xlabel(None)
#plt.ylabel("Particle Push Runtime (s)")
#
#sortfig2 = plt.figure()
#sort_plot2 = sns.barplot(data=df, x='Chip', y='Time Since Last Restore', hue='Sort Order', order=['A100', 'H100', 'MI100', 'MI250', 'MI300A'], hue_order=['Random', 'Standard', 'Strided', 'Tiled Strided'])
#plt.yscale('log')
#plt.xlabel(None)
#plt.ylabel("Particle Push Runtime (s)")
#
#sortfig3, (axtop, axbot) = plt.subplots(ncols=1, nrows=2, sharex=True, height_ratios=[1,3])
#axtop = sns.barplot(ax=axtop, data=df, x='Chip', y='Time Since Last Restore', hue='Sort Order', order=['A100', 'H100', 'MI100', 'MI250', 'MI300A'], hue_order=['Random', 'Standard', 'Strided', 'Tiled Strided'])
#axbot = sns.barplot(ax=axbot, data=df, x='Chip', y='Time Since Last Restore', hue='Sort Order', order=['A100', 'H100', 'MI100', 'MI250', 'MI300A'], hue_order=['Random', 'Standard', 'Strided', 'Tiled Strided'])
#axtop.set_ylim(800, 7500)
#axbot.set_ylim(0, 800)
#axtop.get_xaxis().set_visible(False)
#axbot.get_legend().remove()
#axtop.legend(ncols=2)
#axtop.set_ylabel(None)
#axbot.set_ylabel(None)
#axbot.set_xlabel(None)
#sortfig3.text(0.02, 0.55, "Particle Push Runtime (s)", va="center", rotation="vertical")
#axtop.xaxis.tick_top()
#axbot.xaxis.tick_bottom()


plt.tight_layout()
plt.show()
#fig.savefig(args.output_fname[0])

