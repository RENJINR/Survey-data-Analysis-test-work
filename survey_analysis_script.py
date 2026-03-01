# -*- coding: utf-8 -*-
"""
Supply Chain Resilience Visualization Generator - UPDATED
Author: Your Name
Date: Current Date
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set academic style - UPDATED FOR MODERN MATPLOTLIB
sns.set_style("whitegrid")  # This replaces plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'grid.color': '0.9',  # Light gray grid lines
    'grid.linestyle': '--',
    'grid.linewidth': 0.5
})

# Sample data structure (replace with your actual data loading)
data = {
    'disruptions': {
        'Global Crises': 0.68,
        'Supplier Failures': 0.52,
        'Transport Delays': 0.47,
        'Cybersecurity': 0.31,
        'Natural Disasters': 0.19
    },
    'tech_adoption': {
        'Large': {'AI': 0.62, 'IoT': 0.55, 'Big Data': 0.58, 'Cloud': 0.72},
        'Medium': {'AI': 0.28, 'IoT': 0.34, 'Big Data': 0.41, 'Cloud': 0.53},
        'Small': {'AI': 0.10, 'IoT': 0.18, 'Big Data': 0.22, 'Cloud': 0.31}
    },
    'effectiveness': {
        'Low Tech': [3.1, 2.8, 2.9, 3.4, 3.2, 3.5, 3.0, 3.3, 3.6, 2.9],
        'Single Tech': [3.8, 4.0, 3.6, 3.9, 3.7, 4.1, 3.5, 3.8, 4.2, 3.9],
        'Combo Tech': [4.5, 4.2, 4.7, 4.3, 4.6, 4.4, 4.8, 4.1, 4.5, 4.7]
    },
    'barriers': {
        'High Costs': 0.62,
        'Skill Gaps': 0.45,
        'Integration': 0.38,
        'Resistance': 0.29
    }
}

# =============================================
# 1. Disruption Frequency Plot (Figure 1)
# =============================================
plt.figure(figsize=(8, 4))
disrupt_df = pd.DataFrame.from_dict(data['disruptions'], orient='index', columns=['Frequency'])
disrupt_df = disrupt_df.sort_values('Frequency', ascending=True)

ax = disrupt_df.plot(kind='barh', color='#1f77b4', width=0.7)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
plt.title('Frequency of Supply Chain Disruptions (n=150)')
plt.xlabel('Percentage of Companies Affected')
plt.ylabel('Disruption Type')
plt.tight_layout()
plt.savefig('disruption_frequency.png', bbox_inches='tight')
plt.close()

# =============================================
# 2. Technology Adoption by Company Size (Figure 2)
# =============================================
plt.figure(figsize=(10, 5))
tech_df = pd.DataFrame(data['tech_adoption']).T
tech_df = tech_df[['Cloud', 'Big Data', 'IoT', 'AI']]  # Ordered by adoption

ax = tech_df.plot(kind='bar', width=0.8)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
plt.title('Technology Adoption by Company Size')
plt.ylabel('Adoption Rate')
plt.xlabel('Company Size')
plt.xticks(rotation=0)
plt.legend(title='Technology', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('tech_adoption_by_size.png', bbox_inches='tight')
plt.close()

# =============================================
# 3. Resilience by Technology Profile (Figure 3)
# =============================================
plt.figure(figsize=(8, 5))
effect_data = pd.DataFrame({
    'Low Tech': data['effectiveness']['Low Tech'],
    'Single Tech': data['effectiveness']['Single Tech'],
    'Combo Tech': data['effectiveness']['Combo Tech']
})

# Create boxplot
boxprops = dict(linewidth=1.5, facecolor='white')
medianprops = dict(linewidth=2, color='firebrick')
whiskerprops = dict(linewidth=1.5)
capprops = dict(linewidth=1.5)

bplot = plt.boxplot(
    [effect_data['Low Tech'],
    effect_data['Single Tech'],
    effect_data['Combo Tech']],
    labels=['Low Tech', 'Single Tech', 'Combo Tech'],
    patch_artist=True,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops
)

# Set colors for boxes
colors = ['lightgray', 'skyblue', 'steelblue']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Resilience Score by Technology Adoption Profile')
plt.ylabel('Resilience Score (1-5 Scale)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('resilience_by_tech_profile.png', bbox_inches='tight')
plt.close()

# =============================================
# 4. Implementation Barriers (Figure 4)
# =============================================
plt.figure(figsize=(8, 5))
barrier_df = pd.DataFrame.from_dict(data['barriers'], orient='index', columns=['Percentage'])
barrier_df = barrier_df.sort_values('Percentage', ascending=False)

colors = sns.color_palette("Reds_r", len(barrier_df))  # _r reverses the palette
explode = (0.1, 0, 0, 0)  # Highlight the largest slice

plt.pie(
    barrier_df['Percentage'],
    labels=barrier_df.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    explode=explode,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    textprops={'fontsize': 9}
)

plt.title('Barriers to Technology Implementation')
plt.tight_layout()
plt.savefig('implementation_barriers.png', bbox_inches='tight')
plt.close()

# =============================================
# 5. Additional Trend Plot: AI Implementation Stages
# =============================================
ai_data = {
    'Stage': ['Pilot', 'Partial', 'Full'],
    'Percentage': [0.45, 0.34, 0.21]
}

plt.figure(figsize=(8, 4))
ax = sns.barplot(x='Stage', y='Percentage', data=ai_data,
            order=['Pilot', 'Partial', 'Full'],
            color='#2ca02c', saturation=0.8)

plt.title('AI Implementation Maturity (n=150)')
plt.ylabel('Percentage of Companies')
plt.xlabel('Implementation Stage')
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
plt.tight_layout()
plt.savefig('ai_implementation_stages.png', bbox_inches='tight')
plt.close()

print("All visualizations generated successfully!")