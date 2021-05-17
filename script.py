from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# from scipy.stats import t
# from scipy.optimize import curve_fit
# from scipy.stats.morestats import _add_axis_labels_title

pd.options.mode.chained_assignment = None  # default='warn'


data_dir = '/Users/harperwallace/Dropbox/ENS/LITREV/_writeup/'
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plot_all = True
save_plots = False
show_one_by_one = True

df = pd.read_csv(data_dir + 'Coding - data_grouped.csv')

# drop empty rows
nan_value = float('NaN')
df.replace('', nan_value, inplace=True)
df.dropna(subset = ['HE_short'], inplace=True)

# # sorting HE/harm groups
# df['HE_short'] = df['HE_short'].astype('category')
# df['HE_short'].cat.reorder_categories(np.flip(['AH', 'CH']), inplace=True)

# df['harm_short'] = df['harm_short'].astype('category')
# df['harm_short'].cat.reorder_categories(np.flip(['to_others', 'to_self', 'suicide']), inplace=True)

# df = df.sort_values('end', ascending=False)
# df = df.sort_values(['HE_short', 'harm_short', 'end'], ascending=False)
df = df.sort_values('order')

# calculate standard error in LOG ODDS RATIO
df['se_unadj'] = df.apply(lambda row: (row['ci95_upper_unadj'] - row['ci95_lower_unadj']) / (2 * 1.96 * row['effect_unadj']), axis=1)
df['se_adj'] =   df.apply(lambda row: (row['ci95_upper_adj'] -   row['ci95_lower_adj'])   / (2 * 1.96 * row['effect_adj']),   axis=1)




# # separate hallucination type (CH as subtype of AH)
# df_ah = df
# df_ch = df[df.HE_short == 'CH']

# # separate harm types (suicide as subtype of to_self)
# to_others_ah = df_ah[df_ah.harm_short == 'to_others']
# to_self_ah   = df_ah[(df_ah.harm_short == 'to_self') | (df_ah.harm_short == 'suicide')]
# # suicide_ah   = df_ah[df_ah.harm_short == 'suicide']

# # to_others_ch = df_ch[df_ch.harm_short == 'to_others']
# to_self_ch   = df_ch[(df_ch.harm_short == 'to_self') | (df_ch.harm_short == 'suicide')]
# # suicide_ch   = df_ch[df_ch.harm_short == 'suicide']


###
# to self (AH)
to_plot     = df
to_plot_ch  = to_plot[to_plot.HE_short == 'CH']
n_studies   = len(to_plot)


### forest plot
fig, ax = plt.subplots(figsize=(7, n_studies * 0.3 + 0.9), ncols=2, sharey='all', gridspec_kw={'width_ratios': [1, 1.8]})
marker_scale = 1 / 130

# start cumulative gap at 1 to shift ylabels to start at 1 instead of 0
c_gap = 1
harm_gap_accounted = ''
for i, (study_name, HE_short, harm_short, start_year, end_year, estimated, published, n, effect_unadj, lower_unadj, upper_unadj, effect_adj, lower_adj, upper_adj) in enumerate(zip(to_plot.study_name, to_plot.HE_short, to_plot.harm_short, to_plot.start, to_plot.end, to_plot.estimated, to_plot.published, to_plot.n_total, to_plot.effect_unadj, to_plot.ci95_lower_unadj, to_plot.ci95_upper_unadj, to_plot.effect_adj, to_plot.ci95_lower_adj, to_plot.ci95_upper_adj)):

    if harm_short != 'NASI' and harm_short != harm_gap_accounted:
        if harm_short == 'V':
            gap_shift = 0.5 * (2 if harm_short == 'V' else 1)
            c_gap += gap_shift
            ax[0].axhline(i + c_gap - 0.5 * gap_shift - 0.57, c='k', linewidth=0.5)
            ax[0].axhline(i + c_gap - 0.5 * gap_shift - 0.43, c='k', linewidth=0.5)
        else:
            gap_shift = 0.5
            c_gap += gap_shift
            ax[0].axhline(i + c_gap - 0.5 * gap_shift - 0.5, c='k', linewidth=0.5)
        harm_gap_accounted = harm_short
    
    ax[0].text(0.03, i + c_gap, study_name, va='center')
    ax[0].text(0.62, i + c_gap, r'$*$' if HE_short == 'CH' else '', va='center')
    ax[0].text(0.74, i + c_gap, ('   ' if (harm_short == 'NSSI' or harm_short == 'SA') else '') + harm_short, va='center')
    
    ax[1].errorbar(effect_unadj, i + c_gap - 0.15, xerr=np.array([[lower_unadj, upper_unadj]]).T, marker='s', color='k',         markersize=n*marker_scale, capsize=3, zorder=10)
    ax[1].errorbar(effect_adj,   i + c_gap + 0.15, xerr=np.array([[lower_adj,   upper_adj]]).T,   marker='s', color='lightgray', markersize=n*marker_scale, capsize=3, zorder=5)


ax[0].axis('off')
# ax[0].get_xaxis().set_visible(False)
# ax[0].set_ylabel('Study')
# ax[0].set_yticks(np.linspace(1, n_studies, n_studies))
ax[0].set_yticks([])
ax[0].invert_yaxis()

ax[1].set_xlabel('Odds ratio (CI$_{95}$)')
# make odds ratio log scale, but keep tick labels decimal
ax[1].set_xscale('log')
ax[1].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax[1].axvline(1, color='k', linestyle=':', zorder=0)

fig.tight_layout()
fig.savefig(data_dir + '_plots/forest.png', dpi=600)

plt.close(fig)

