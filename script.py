import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from scipy.stats import t
from scipy.optimize import curve_fit

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

# sorting HE/harm groups
df['HE_short'] = df['HE_short'].astype('category')
df['HE_short'].cat.reorder_categories(np.flip(['AH', 'CH']), inplace=True)

df['harm_short'] = df['harm_short'].astype('category')
df['harm_short'].cat.reorder_categories(np.flip(['to_others', 'to_self', 'suicide']), inplace=True)

# df = df.sort_values('end', ascending=False)
df = df.sort_values(['HE_short', 'harm_short', 'end'], ascending=False)

# calculate standard error
df['se_unadj'] = df.apply(lambda row: (np.log(row['ci95_upper_unadj'] - np.log(row['ci95_lower_unadj'])) / (2 * 1.96)), axis=1)
df['se_adj'] =   df.apply(lambda row: (np.log(row['ci95_upper_adj'] -   np.log(row['ci95_lower_adj']))   / (2 * 1.96)), axis=1)



# separate hallucination type (CH as subtype of AH)
df_ah = df
df_ch = df[df['HE_short'] == 'CH']

# separate harm types (suicide as subtype of to_self)
to_others_ah = df_ah[df_ah['harm_short'] == 'to_others']
to_self_ah   = df_ah[(df_ah['harm_short'] == 'to_self') | (df_ah['harm_short'] == 'suicide')]
suicide_ah   = df_ah[df_ah['harm_short'] == 'suicide']

to_others_ch = df_ch[df_ch['harm_short'] == 'to_others']
to_self_ch   = df_ch[(df_ch['harm_short'] == 'to_self') | (df_ch['harm_short'] == 'suicide')]
suicide_ch   = df_ch[df_ch['harm_short'] == 'suicide']



to_plot = to_self_ah
# to_plot = to_others_ah
# to_plot = to_self_ah


n_studies = len(to_plot)


### forest plot
fig, ax = plt.subplots(figsize=(7, n_studies * 0.3 + 0.9), ncols=3, sharey='all', gridspec_kw={'width_ratios': [1, 1.6, 0.2]})
marker_scale = 1 / 130

for i, (start_year, end_year, estimated, published, n, effect_unadj, lower_unadj, upper_unadj, effect_adj, lower_adj, upper_adj) in enumerate(zip(to_plot.start, to_plot.end, to_plot.estimated, to_plot.published, to_plot.n_total, to_plot.effect_unadj, to_plot.ci95_lower_unadj, to_plot.ci95_upper_unadj, to_plot.effect_adj, to_plot.ci95_lower_adj, to_plot.ci95_upper_adj)):
    
    study_years_estimated = (estimated != estimated)
    ax[0].plot((start_year, end_year), (i + 1, i + 1), '-', linewidth=1, color='black' if study_years_estimated else 'lightgray')
    ax[0].plot(published, i + 1, 'wo', mec='k', markersize=4)

    ax[1].errorbar(effect_unadj, i + 1 - 0.15, xerr=np.array([[lower_unadj, upper_unadj]]).T, marker='s', color='lightgray', markersize=n*marker_scale, capsize=3, zorder=5)
    ax[1].errorbar(effect_adj,   i + 1 + 0.15, xerr=np.array([[lower_adj,   upper_adj]]).T,   marker='s', color='k',         markersize=n*marker_scale, capsize=3, zorder=10)

# if to_self; no need for to_self distinction for to_others or suicide
ax[2].pcolor(np.array([to_plot['HE_short'].values == 'CH', to_plot['harm_short'].values == 'to_self']).T)

ax[0].set_xlabel('Year')
ax[0].set_ylabel('Study')
ax[0].set_yticks(np.linspace(1, n_studies, n_studies))

ax[1].set_xlabel('Odds ratio (log scale)')

# make odds ratio log scale, but keep tick labels decimal
ax[1].set_xscale('log')
ax[1].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax[1].axvline(1, color='k', linestyle=':', zorder=0)

fig.tight_layout()
fig.savefig(data_dir + '_plots/figa.png', dpi=300)

plt.close(fig)


### funnel plot
complete = to_plot[(~np.isnan(to_plot.effect_unadj)) & (~np.isnan(to_plot.se_unadj))]
to_plot = complete
print(to_plot)

X = to_plot.effect_unadj.values
Y = 1 / to_plot.se_unadj.values
weights = to_plot.n_total

fig, ax = plt.subplots()
marker_scale = 1 / 8

mean_odds_ratio = np.mean(X)
ax.scatter(X, Y, s=to_plot.n_total*marker_scale, c='k', zorder=10)


# regression
func = lambda x, m, b : m * x + b

fit_params, cov = curve_fit(func, np.log(X), Y, sigma=weights)
x_pred_lims = (-0.3, 2.8)
log_x_pred = np.linspace(*x_pred_lims)
y_pred = func(log_x_pred, *fit_params)

ax.plot(np.exp(log_x_pred), y_pred, zorder=5, c='k')

# unweighted r**2
residuals = Y - func(np.log(X), *fit_params)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Y - np.mean(Y))**2)
r_squared = 1 - (ss_res / ss_tot)

custom_lines = [ Line2D([0], [0], color='lightgray', linestyle='--'),
                 Line2D([0], [0], color='k'),
                 Line2D([0], [0], color='red')]
ax.legend(custom_lines, [ 'mean (OR$=%.2f$)' % mean_odds_ratio,
                          'linear fit ($r^2=%.2f$)' % r_squared,
                          'fitted intercept (95% C.I.)'])

# unweighted standard error in intercept
n = len(X)
n_p = 2
if n > n_p:
    # x_squares = np.sum(X**2) / np.sum((X - np.mean(X))**2)
    # se_alpha = (ss_res * x_squares / (n * (n - 2)))**0.5
    # t_val = lambda df, conf : t.ppf((1 + conf) / 2., df)
    # ci_error_alpha = se_alpha * t_val(n - 2, 0.95)
    # ci = (fit_params[1] - ci_error_alpha, fit_params[1] + ci_error_alpha)
    # ax.errorbar(1, fit_params[1], ci_error_alpha, c='red', capsize=3)

    X_with_intercept = np.empty(shape=(n, n_p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:n_p] = X
    
    y_hat = func(np.log(X), *fit_params)
    residuals = Y - y_hat
    residual_sum_of_squares = residuals.T @ residuals
    sigma_squared_hat = residual_sum_of_squares / (n - n_p)
    var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
    for p_ in range(n_p):
        standard_error = var_beta_hat[p_, p_] ** 0.5
        print(f"SE(beta_hat[{p_}]): {standard_error}")
else:
    print('n too small for se_alpha')
    


ax.axvline(1, color='k', linestyle=':', zorder=0)
ax.axvline(mean_odds_ratio, c='lightgray', linestyle='--', zorder=0)

ax.set_xlabel('Odds ratio (log scale)')
ax.set_ylabel('Inverse standard error')

# make odds ratio log scale, but keep tick labels decimal
ax.set_xscale('log')
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

ax.set_xlim(xmin=np.exp(x_pred_lims[0]), xmax=np.exp(x_pred_lims[1]))
# ax.set_ylim(ymin=0, ymax=4.3)

fig.tight_layout()
fig.savefig(data_dir + '_plots/figb.png', dpi=300)
plt.close(fig)

plt.show()
