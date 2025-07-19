import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#%%
# Parameters
N_prior = 100
N_posterior = 10
x = np.arange(1, N_prior + 1)

# Data points
y_prior = np.ones(N_prior)
x_q_post = x[:N_posterior]
y_q_post = np.full(N_posterior, 2)
x_swe1 = x_q_post
y_swe1 = np.full(N_posterior, 3)
np.random.seed(42)
# x_swe2 = np.random.choice(x, N_posterior, replace=False)
x_swe2 = np.arange(6, N_prior + 1,step = (N_prior - N_posterior) / (N_posterior - 1))
y_swe2 = np.full(N_posterior, 4)

# Medians
median_q_post = np.median(x_q_post)
median_swe1 = np.median(x_swe1)
median_swe2 = (N_prior+1)/2# np.median(x_swe2)

# Plot
fig, ax = plt.subplots(figsize=(6,3))

# Set custom font
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['font.size'] = 12
color = (0.2627450980392157, 0.6980392156862745, 0.5176470588235295, 1.0)
# Plotting with larger markers
ax.scatter(x, y_prior, color='lightgray', s=50, zorder=1,edgecolors='black')
ax.scatter(x_q_post, y_q_post, color=color, s=100, edgecolors='black', zorder=2)
ax.scatter(x_swe1, y_swe1, color=color, s=100, edgecolors='black', zorder=3)
ax.scatter(x_swe2, y_swe2, color=color, s=100, edgecolors='black', zorder=4)

# Text annotations
# ax.text(N_prior * 0.15, 2.9, "Perfect constraint", color='black', fontsize=10)
# ax.text(N_prior * 0.55, 3.9, "No constraint", color='black', fontsize=10)

# Medians and dashed lines
# ax.scatter(median_q_post, 2, color='black', s=80, marker='|', zorder=5)
# ax.scatter(median_swe1, 3, color='black', s=80, marker='|', zorder=5)
# ax.scatter(median_swe2, 4, color='black', s=80, marker='|', zorder=5)
ax.plot(median_q_post, 2, color='black', marker  ='|', markersize=15,
        linewidth = 0,markeredgewidth=2,
         label='= Median Rank',zorder = 5)
ax.plot(median_swe1, 3, color='black', marker  ='|', markersize=15,
        linewidth = 0,markeredgewidth=2,zorder=5)
ax.plot(median_swe2, 4, color='black', marker  ='|', markersize=15,
        linewidth = 0,markeredgewidth=2,zorder=5)
# ax.vlines(median_q_post, 2, 1.5, linestyles='dashed', color='black')
# ax.vlines(median_swe1, 3, 2.5, linestyles='dashed', color='black')
# ax.vlines(median_swe2, 4, 2.5, linestyles='dashed', color='black',
#           label = 'Median Rank')
ax.legend(loc = 'right', fontsize = 10, frameon = False,draggable = True)

# Axis settings
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(['All Runs', 'Q metric',
                    # 'SWE posterior \n Metric 1', 'SWE posterior \n Metric 2'])
                     f"SWE metric 1 \n (e.g. Melt error)", 
                     f"SWE metric 2 \n (e.g. Snowfall Error)"])
ax.set_ylim(4.5, 0.5)  # Flip y-axis

ytick_colors = ['gray', 'blue', 'blue', 'blue']
ytick_labels = ['All Runs', 'Q metric', 'SWE metric 1', 'SWE metric 2']

# Empty tick labels first
# ax.set_yticklabels([])

# Add colored text labels manually
# for i, (label, color) in enumerate(zip(ytick_labels, ytick_colors)):
#     ax.text(-5, i+1, label, color=color, va='center', fontsize=10)

# Add legend elements for prior/posterior
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
           markeredgecolor='black', markersize=8, label='Prior'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
           markeredgecolor='black', markersize=8, label='Posterior'),
    Line2D([0], [0], marker='|', color='black', markersize=10, 
           markeredgewidth=2, linewidth = 0,label='Median Rank')
]

# Create a new combined legend
ax.legend(handles=legend_elements, loc='right', 
          fontsize=10, frameon=False, draggable=True)

# Custom x-ticks
# xticks = [1, (N_posterior+1) / 2, N_posterior, (N_prior+1) / 2, N_prior]
# xtick_labels = ['$1$', r'$0.5*(N_{posterior}+1)$', r'$N_{posterior}$', r'$0.5*(N_{prior}+1)$', r'$N_{prior}$']


# xticks = [1, N_posterior,  N_prior]
# xtick_labels = ['$1$', '$N_{posterior}$', '$N_{prior}$']

xticks = [1,  N_prior]
xtick_labels = ['$1$', '$N_{prior}$']

ax.set_xticks(xticks)
ax.set_xticklabels([''] * len(xticks))  # Empty labels for bottom axis
# ax.set_xlabel('Model Run Rank')
ax.set_xlim(0, N_prior + 1)

# Create a twin axis for the top
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(xticks)
ax_top.set_xticklabels(xtick_labels)

ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.spines['bottom'].set_visible(False)
ax_top.spines['left'].set_visible(False)
ax_top.grid()
ax_top.set_xlabel('Model Run Rank')


# ax.set_xticks(xticks)
# # ax.set_xticklabels(xtick_labels,rotation = 45, ha='right')
# ax.set_xticklabels(xtick_labels)
# ax.set_xlabel('Model Run Rank')
# ax.grid()
# ax.set_xlim(0, N_prior + 1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# ax.set_title('Posterior Rank Illustration')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join("/home/pwiersma/scratch/Figures", 'Posterior_Rank_Illustration.svg'), dpi=300, bbox_inches='tight')
plt.show()
