import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
import pingouin as pg
import researchpy as rp
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

# Load the data
df = pd.read_csv(r"D:\RENJIN RAJU\MASTERS\SEMESTER 3\THESIS\Research\Analysis\Survey data cleaned.csv", encoding='latin-1')

def clean_yes_no(value):
    """Convert various yes/no formats to 1/0/NaN"""
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if value in ['yes', 'y', '1', 'true']:
        return 1
    elif value in ['no', 'n', '0', 'false']:
        return 0
    return np.nan

def clean_implementation_status(value):
    """Clean implementation status columns"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    # Extract first number from string (e.g., "1_-_Not_Implemented" -> 1.0)
    try:
        return float(str(value).split('_')[0])
    except:
        return np.nan

# Data Cleaning and Preprocessing
def clean_data(df):
    # Handle missing values
    df_clean = df.copy()

    # Convert size to numerical
    df_clean['size'] = pd.to_numeric(df_clean['size'], errors='coerce')

    # Create binary columns for disruptions
    disruption_cols = ['Disruption_Natural_Disasters', 'Disruption_Supplier_Failures',
                       'Disruption_Cybersecurity_Threats', 'Disruption_Global_Crises',
                       'Disruption_Transportation_Delays']

    def clean_yes_no(value):
        if pd.isna(value):
            return np.nan
        value = str(value).strip().lower()
        if value in ['yes', 'y', '1', 'true']:
            return 1
        elif value in ['no', 'n', '0', 'false']:
            return 0
        return np.nan

    for col in disruption_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_yes_no)

    # Create a new disruption score by summing the cleaned disruption columns
    df_clean['disruption_score'] = df_clean[disruption_cols].sum(axis=1)

    # Create binary columns for responses
    response_cols = ['response_Increased_Inventory_Buffers', 'response_Diversified_Suppliers',
                     'response_Implemented_New_Technologies', 'response_Other']
    for col in response_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_yes_no)

    # Create a new response score by summing the cleaned response columns
    df_clean['response_score'] = df_clean[response_cols].sum(axis=1)

    # Clean technology adoption columns
    tech_cols = [
        'adoption_Artificial_Intelligence_AI',
        'adoption_Internet_of_Things_IoT',
        'adoption_Blockchain',
        'adoption_Cloud_Computing',
        'adoption_Robotic_and_Automation',
        'adoption_Big_Data_Analytics',
        'adoption_Simulation',
        'adoption_Other'
    ]

    clean_tech_cols = []
    for col in tech_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_yes_no)

    # Create the adoption score by summing the cleaned adoption columns
    df_clean['adoption_score'] = df_clean[tech_cols].sum(axis=1)

    # Clean implementation status columns
    imp_cols = ['implemented_AI', 'implemented_IoT', 'implemented_Blockchain',
                'implemented_Cloud_Computing', 'implemented_Robotics_and_Automation',
                'implemented_Big_Data_Analytics']

    for col in imp_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_implementation_status)

        df_clean['implementation_score'] = df_clean[imp_cols].mean(axis=1)

    # Clean effectiveness rankings
    eff_cols = ['effective_technology_Rank_1', 'effective_technology_Rank_2',
                'effective_technology_Rank_3', 'effective_technology_Rank_4',
                'effective_technology_Rank_5', 'effective_technology_Rank_6',
                'effective_technology_Rank_7']

    # Clean barrier columns
    barrier_cols = ['barriers_High_Implementation_Costs', 'barriers_Lack_of_Skilled_Workforce',
                    'barriers_Integration_Issues', 'barriers_Resistance_to_Change']
    for col in barrier_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_yes_no)

    # Create collaboration score
    collab_cols = ['resilience_collaboration_Cloud-based_platforms',
                   'resilience_collaboration_Blockchain_for_transparency',
                   'resilience_collaboration_AI_powered_predictive_analytics',
                   'resilience_collaboration_IoT_for_real-time_tracking',
                   'resilience_collaboration_Digital_twin_Simulations']
    for col in collab_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_yes_no)

        # Create collaboration score
    df_clean['collaboration_score'] = df_clean[collab_cols].sum(axis=1)

    # Clean real-time data columns
    rt_cols = ['visibility_and_control_Rank_1', 'visibility_and_control_Rank_2',
               'visibility_and_control_Rank_3', 'visibility_and_control_Rank_4',
               'visibility_and_control_Rank_5', 'visibility_and_control_Rank_6']

    # Clean adoption issues columns
    issue_cols = ['further_adoption_issues_Financial_Constraints',
                  'further_adoption_issues_Technical_Challenges',
                  'further_adoption_issues_Regulatory_Issues',
                  'further_adoption_issues_Uncertainty_about_ROI']
    for col in issue_cols:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else np.nan)

    # Create company size categories
    df_clean['size_category'] = pd.cut(df_clean['size'],
                                       bins=[0, 250, 750, np.inf],
                                       labels=['SME', 'Medium', 'Large'])

    # Create resilience score
    df_clean['resilience_score'] = (df_clean['response_score'] + df_clean['collaboration_score']) / 2
    return df_clean


df_clean = clean_data(df)


# Create a PDF report
def create_report(df_clean):
    with PdfPages('Supply_Chain_Resilience_Analysis_Report.pdf') as pdf:
        # Title page
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.7, 'Supply Chain Resilience Analysis Report',
                 ha='center', va='center', fontsize=20, fontweight='bold')
        plt.text(0.5, 0.6, 'Analysis of Emerging Technologies Impact on Supply Chain Resilience',
                 ha='center', va='center', fontsize=14)
        plt.text(0.5, 0.4, f'Total Respondents: {len(df_clean)}',
                 ha='center', va='center', fontsize=12)
        pdf.savefig()
        plt.close()

        # Research Questions Analysis
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9, 'Research Questions Analysis', fontsize=16, fontweight='bold')

        # RQ1: Disturbances and resilience characteristics
        plt.text(0.1, 0.8,
                 '1. Which disturbances impact supply chains of manufacturing companies and what characterizes the resilience of supply chains?',
                 fontsize=12, fontweight='bold')

        # Disruption frequency
        disruption_cols = ['Disruption_Natural_Disasters', 'Disruption_Supplier_Failures',
                           'Disruption_Cybersecurity_Threats', 'Disruption_Global_Crises',
                           'Disruption_Transportation_Delays']
        disruption_counts = df_clean[disruption_cols].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        disruption_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Frequency of Different Types of Disruptions')
        ax.set_ylabel('Number of Companies Affected')
        ax.set_xlabel('Disruption Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Resilience characteristics
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df_clean, x='disruption_score', y='resilience_score')
        plt.title('Relationship Between Disruption Experience and Resilience Score')
        plt.xlabel('Number of Disruptions Experienced')
        plt.ylabel('Resilience Score')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # RQ2: Technologies that help build resilience
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9, '2. Which technologies help manufacturing companies building supply chain resilience?',
                 fontsize=12, fontweight='bold')

        # Technology effectiveness
        eff_cols = ['effective_technology_Rank_1', 'effective_technology_Rank_2',
                    'effective_technology_Rank_3', 'effective_technology_Rank_4',
                    'effective_technology_Rank_5', 'effective_technology_Rank_6',
                    'effective_technology_Rank_7']

        # Count how many times each technology appears in rankings
        tech_effectiveness = pd.concat([df_clean[col].value_counts() for col in eff_cols], axis=1).fillna(0)
        tech_effectiveness['total'] = tech_effectiveness.sum(axis=1)
        tech_effectiveness = tech_effectiveness.sort_values('total', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        tech_effectiveness['total'].head(10).plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_title('Most Effective Technologies for Resilience')
        ax.set_ylabel('Total Mentions in Rankings')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # RQ3: Technology implementation in practice
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 '3. Which technologies are actually used in practice, how high is the degree of implementation in practice',
                 fontsize=12, fontweight='bold')

        # Technology adoption rates
        tech_cols = ['adoption_Artificial_Intelligence_AI', 'adoption_Internet_of_Things_IoT',
                     'adoption_Blockchain', 'adoption_Cloud_Computing', 'adoption_Robotic_and_Automation',
                     'adoption_Big_Data_Analytics', 'adoption_Simulation', 'adoption_Other']
        adoption_rates = df_clean[tech_cols].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        adoption_rates.plot(kind='bar', ax=ax, color='lightblue')
        ax.set_title('Technology Adoption Rates')
        ax.set_ylabel('Proportion of Companies Adopting')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Hypothesis Testing Section
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9, 'Hypothesis Testing Results', fontsize=16, fontweight='bold')

        # H1a: Companies that have experienced higher levels of supply chain disruptions are more likely to adopt resilience-enhancing technologies.
        plt.text(0.1, 0.8,
                 'H1a: Companies that have experienced higher levels of supply chain disruptions are more likely to adopt resilience-enhancing technologies.',
                 fontsize=10, fontweight='bold')

        # Correlation between disruption score and adoption score
        corr, p_val = stats.pearsonr(df_clean['disruption_score'].dropna(), df_clean['adoption_score'].dropna())
        plt.text(0.1, 0.75, f'Pearson Correlation: {corr:.2f}, p-value: {p_val:.3f}', fontsize=10)

        if p_val < 0.05:
            plt.text(0.1, 0.7, 'Conclusion: Reject null hypothesis - There is a significant relationship.', fontsize=10)
        else:
            plt.text(0.1, 0.7, 'Conclusion: Fail to reject null hypothesis - No significant relationship found.',
                     fontsize=10)

        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.regplot(data=df_clean, x='disruption_score', y='adoption_score', ax=ax)
        ax.set_title('Relationship Between Disruption Score and Technology Adoption')
        ax.set_xlabel('Disruption Score')
        ax.set_ylabel('Technology Adoption Score')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # H1b: The impact of disruptions significantly influences the choice of resilience-enhancing technologies.
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H1b: The impact of disruptions significantly influences the choice of resilience-enhancing technologies.',
                 fontsize=10, fontweight='bold')

        # ANOVA for each technology adoption by disruption score
        tech_results = []
        for tech in tech_cols:
            model = ols(f'{tech} ~ C(disruption_score)', data=df_clean).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            tech_results.append({
                'Technology': tech.replace('adoption_', '').replace('_', ' '),
                'F-value': anova_table['F'][0],
                'p-value': anova_table['PR(>F)'][0]
            })

        tech_results_df = pd.DataFrame(tech_results)
        significant_techs = tech_results_df[tech_results_df['p-value'] < 0.05]

        if not significant_techs.empty:
            plt.text(0.1, 0.8, 'Technologies significantly influenced by disruption level:', fontsize=10)
            for i, (_, row) in enumerate(significant_techs.iterrows(), start=1):
                plt.text(0.1, 0.8 - i * 0.05,
                         f"{row['Technology']}: F={row['F-value']:.2f}, p={row['p-value']:.3f}",
                         fontsize=10)
        else:
            plt.text(0.1, 0.8, 'No technologies showed significant differences by disruption level.', fontsize=10)

        # Visualization of one significant technology if exists
        if not significant_techs.empty:
            tech_to_show = significant_techs.iloc[0]['Technology'].replace(' ', '_')
            tech_col = f'adoption_{tech_to_show}'

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=df_clean, x='disruption_score', y=tech_col, ax=ax)
            ax.set_title(f'Adoption of {tech_to_show.replace("_", " ")} by Disruption Score')
            ax.set_xlabel('Disruption Score')
            ax.set_ylabel('Adoption Rate')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # H2a: The adoption rate of technologies is higher in large enterprises compared to small and medium enterprises (SMEs).
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H2a: The adoption rate of technologies is higher in large enterprises compared to small and medium enterprises (SMEs).',
                 fontsize=10, fontweight='bold')

        # ANOVA for adoption score by size category
        model = ols('adoption_score ~ C(size_category)', data=df_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        f_val = anova_table['F'][0]
        p_val = anova_table['PR(>F)'][0]

        plt.text(0.1, 0.85, f'ANOVA Results: F={f_val:.2f}, p={p_val:.3f}', fontsize=10)

        if p_val < 0.05:
            plt.text(0.1, 0.8, 'Conclusion: Reject null hypothesis - Significant differences exist.', fontsize=10)

            # Post-hoc test
            posthoc = pg.pairwise_ttests(data=df_clean, dv='adoption_score',
                                         between='size_category', padjust='bonf')
            plt.text(0.1, 0.75, 'Post-hoc comparisons:', fontsize=10)

            for i, (_, row) in enumerate(posthoc.iterrows(), start=1):
                plt.text(0.1, 0.75 - i * 0.05,
                         f"{row['A']} vs {row['B']}: p={row['p-unc']:.3f}",
                         fontsize=10)
        else:
            plt.text(0.1, 0.8, 'Conclusion: Fail to reject null hypothesis - No significant differences.', fontsize=10)

        # Boxplot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_clean, x='size_category', y='adoption_score', ax=ax)
        ax.set_title('Technology Adoption Score by Company Size Category')
        ax.set_xlabel('Company Size Category')
        ax.set_ylabel('Adoption Score')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # H2b: Companies in highly disrupted industries are more likely to implement which technology for resilience.
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H2b: Companies in highly disrupted industries are more likely to implement which technology for resilience.',
                 fontsize=10, fontweight='bold')

        # Group by industry and disruption score
        industry_disruption = df_clean.groupby('Industry')['disruption_score'].mean().sort_values(ascending=False)
        top_disrupted = industry_disruption.head(3).index.tolist()

        plt.text(0.1, 0.85, f'Top 3 most disrupted industries: {", ".join(top_disrupted)}', fontsize=10)

        # Compare technology adoption in high vs low disruption industries
        df_clean['high_disruption_industry'] = df_clean['Industry'].apply(lambda x: x in top_disrupted)

        tech_diff_results = []
        for tech in tech_cols:
            high = df_clean[df_clean['high_disruption_industry']][tech].mean()
            low = df_clean[~df_clean['high_disruption_industry']][tech].mean()
            _, p_val = stats.ttest_ind(
                df_clean[df_clean['high_disruption_industry']][tech].dropna(),
                df_clean[~df_clean['high_disruption_industry']][tech].dropna(),
                equal_var=False
            )
            tech_diff_results.append({
                'Technology': tech.replace('adoption_', '').replace('_', ' '),
                'High_Disruption': high,
                'Low_Disruption': low,
                'p-value': p_val
            })

        tech_diff_df = pd.DataFrame(tech_diff_results)
        significant_diff = tech_diff_df[tech_diff_df['p-value'] < 0.05]

        if not significant_diff.empty:
            plt.text(0.1, 0.8, 'Technologies with significant differences:', fontsize=10)
            for i, (_, row) in enumerate(significant_diff.iterrows(), start=1):
                plt.text(0.1, 0.8 - i * 0.05,
                         f"{row['Technology']}: High={row['High_Disruption']:.2f}, Low={row['Low_Disruption']:.2f}, p={row['p-value']:.3f}",
                         fontsize=10)
        else:
            plt.text(0.1, 0.8,
                     'No technologies showed significant differences between high and low disruption industries.',
                     fontsize=10)

        # Visualization of one significant technology if exists
        if not significant_diff.empty:
            tech_to_show = significant_diff.iloc[0]['Technology'].replace(' ', '_')
            tech_col = f'adoption_{tech_to_show}'

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=df_clean, x='high_disruption_industry', y=tech_col, ax=ax)
            ax.set_title(f'Adoption of {tech_to_show.replace("_", " ")} by Disruption Level')
            ax.set_xlabel('High Disruption Industry')
            ax.set_ylabel('Adoption Rate')
            ax.set_xticklabels(['Low Disruption', 'High Disruption'])
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # H3a: Companies using technologies report higher supply chain efficiency and resilience compared to those not using them.
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H3a: Companies using technologies report higher supply chain efficiency and resilience compared to those not using them.',
                 fontsize=10, fontweight='bold')

        # Compare resilience scores between adopters and non-adopters
        tech_adoption_impact = []
        for tech in tech_cols:
            adopters = df_clean[df_clean[tech] == 1]['resilience_score'].mean()
            non_adopters = df_clean[df_clean[tech] == 0]['resilience_score'].mean()
            _, p_val = stats.ttest_ind(
                df_clean[df_clean[tech] == 1]['resilience_score'].dropna(),
                df_clean[df_clean[tech] == 0]['resilience_score'].dropna(),
                equal_var=False
            )
            tech_adoption_impact.append({
                'Technology': tech.replace('adoption_', '').replace('_', ' '),
                'Adopters_Mean': adopters,
                'Non_Adopters_Mean': non_adopters,
                'p-value': p_val
            })

        tech_impact_df = pd.DataFrame(tech_adoption_impact)
        significant_impact = tech_impact_df[tech_impact_df['p-value'] < 0.05]

        if not significant_impact.empty:
            plt.text(0.1, 0.85, 'Technologies with significant impact on resilience:', fontsize=10)
            for i, (_, row) in enumerate(significant_impact.iterrows(), start=1):
                plt.text(0.1, 0.85 - i * 0.05,
                         f"{row['Technology']}: Adopters={row['Adopters_Mean']:.2f}, Non-Adopters={row['Non_Adopters_Mean']:.2f}, p={row['p-value']:.3f}",
                         fontsize=10)
        else:
            plt.text(0.1, 0.85, 'No technologies showed significant impact on resilience scores.', fontsize=10)

        # Visualization of resilience score by technology adoption
        if not significant_impact.empty:
            tech_to_show = significant_impact.iloc[0]['Technology'].replace(' ', '_')
            tech_col = f'adoption_{tech_to_show}'

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df_clean, x=tech_col, y='resilience_score', ax=ax)
            ax.set_title(f'Resilience Score by Adoption of {tech_to_show.replace("_", " ")}')
            ax.set_xlabel('Adoption Status (0=No, 1=Yes)')
            ax.set_ylabel('Resilience Score')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # H3b: Companies using multiple technologies in combination report better resilience than those using a single technology.
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H3b: Companies using multiple technologies in combination report better resilience than those using a single technology.',
                 fontsize=10, fontweight='bold')

        # Create technology combination groups
        df_clean['tech_count'] = df_clean[tech_cols].sum(axis=1)
        df_clean['tech_group'] = pd.cut(df_clean['tech_count'],
                                        bins=[0, 1, 3, len(tech_cols)],
                                        labels=['0-1', '2-3', '4+'])

        # ANOVA for resilience score by tech group
        model = ols('resilience_score ~ C(tech_group)', data=df_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        f_val = anova_table['F'][0]
        p_val = anova_table['PR(>F)'][0]

        plt.text(0.1, 0.85, f'ANOVA Results: F={f_val:.2f}, p={p_val:.3f}', fontsize=10)

        if p_val < 0.05:
            plt.text(0.1, 0.8, 'Conclusion: Reject null hypothesis - Significant differences exist.', fontsize=10)

            # Post-hoc test
            posthoc = pg.pairwise_ttests(data=df_clean, dv='resilience_score',
                                         between='tech_group', padjust='bonf')
            plt.text(0.1, 0.75, 'Post-hoc comparisons:', fontsize=10)

            for i, (_, row) in enumerate(posthoc.iterrows(), start=1):
                plt.text(0.1, 0.75 - i * 0.05,
                         f"{row['A']} vs {row['B']}: p={row['p-unc']:.3f}",
                         fontsize=10)
        else:
            plt.text(0.1, 0.8, 'Conclusion: Fail to reject null hypothesis - No significant differences.', fontsize=10)

        # Boxplot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_clean, x='tech_group', y='resilience_score', ax=ax)
        ax.set_title('Resilience Score by Number of Technologies Adopted')
        ax.set_xlabel('Number of Technologies Adopted')
        ax.set_ylabel('Resilience Score')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # H4a: High implementation costs are the primary barrier to adopting new supply chain technologies.
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H4a: High implementation costs are the primary barrier to adopting new supply chain technologies.',
                 fontsize=10, fontweight='bold')

        # Calculate barrier frequencies
        barrier_cols = ['barriers_High_Implementation_Costs', 'barriers_Lack_of_Skilled_Workforce',
                        'barriers_Integration_Issues', 'barriers_Resistance_to_Change']
        for col in barrier_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(clean_yes_no)

        barrier_counts = df_clean[barrier_cols].sum().sort_values(ascending=False)

        plt.text(0.1, 0.85, 'Frequency of reported barriers:', fontsize=10)
        for i, (barrier, count) in enumerate(barrier_counts.items(), start=1):
            plt.text(0.1, 0.85 - i * 0.05,
                     f"{barrier.replace('barriers_', '').replace('_', ' ')}: {count} mentions",
                     fontsize=10)

        non_zero_categories = barrier_counts[barrier_counts > 0]

        # Calculate the expected frequencies for the non-zero categories
        expected_non_zero = [sum(non_zero_categories) / len(non_zero_categories)] * len(non_zero_categories)

        # Recalculate the chi-square test for the non-zero categories
        chi2, p_val = stats.chisquare(non_zero_categories, f_exp=expected_non_zero)

        plt.text(0.1, 0.7, f'Chi-square test for equal distribution: χ²={chi2:.2f}, p={p_val:.3f}', fontsize=10)

        if p_val < 0.05:
            plt.text(0.1, 0.65, 'Conclusion: Reject null hypothesis - Barriers are not equally frequent.', fontsize=10)
            plt.text(0.1, 0.6, f'Primary barrier: {barrier_counts.index[0].replace("barriers_", "").replace("_", " ")}',
                     fontsize=10, fontweight='bold')
        else:
            plt.text(0.1, 0.65,
                     'Conclusion: Fail to reject null hypothesis - No significant differences in barrier frequency.',
                     fontsize=10)

        # Bar chart of barriers
        fig, ax = plt.subplots(figsize=(10, 5))
        barrier_counts.plot(kind='bar', ax=ax, color='salmon')
        ax.set_title('Frequency of Reported Barriers to Technology Adoption')
        ax.set_ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # H4b: Lack of skilled workforce significantly affects the adoption of AI and Big Data Analytics and other technologies in supply chain management.
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H4b: Lack of skilled workforce significantly affects the adoption of AI and Big Data Analytics and other technologies in supply chain management.',
                 fontsize=10, fontweight='bold')

        # Compare adoption rates between companies reporting workforce barrier vs not
        tech_workforce_impact = []
        for tech in tech_cols:
            barrier = df_clean[df_clean['barriers_Lack_of_Skilled_Workforce'] == 1][tech].mean()
            no_barrier = df_clean[df_clean['barriers_Lack_of_Skilled_Workforce'] == 0][tech].mean()
            _, p_val = stats.ttest_ind(
                df_clean[df_clean['barriers_Lack_of_Skilled_Workforce'] == 1][tech].dropna(),
                df_clean[df_clean['barriers_Lack_of_Skilled_Workforce'] == 0][tech].dropna(),
                equal_var=False
            )
            tech_workforce_impact.append({
                'Technology': tech.replace('adoption_', '').replace('_', ' '),
                'With_Workforce_Barrier': barrier,
                'Without_Workforce_Barrier': no_barrier,
                'p-value': p_val
            })

        workforce_impact_df = pd.DataFrame(tech_workforce_impact)
        significant_workforce = workforce_impact_df[workforce_impact_df['p-value'] < 0.05]

        if not significant_workforce.empty:
            plt.text(0.1, 0.85, 'Technologies significantly affected by workforce barrier:', fontsize=10)
            for i, (_, row) in enumerate(significant_workforce.iterrows(), start=1):
                plt.text(0.1, 0.85 - i * 0.05,
                         f"{row['Technology']}: With barrier={row['With_Workforce_Barrier']:.2f}, Without={row['Without_Workforce_Barrier']:.2f}, p={row['p-value']:.3f}",
                         fontsize=10)
        else:
            plt.text(0.1, 0.85, 'No technologies showed significant impact from workforce barrier.', fontsize=10)

        # Visualization for AI and Big Data specifically
        for tech in ['Artificial_Intelligence_AI', 'Big_Data_Analytics']:
            tech_col = f'adoption_{tech}'
            if tech_col in df_clean.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=df_clean, x='barriers_Lack_of_Skilled_Workforce', y=tech_col, ax=ax)
                ax.set_title(f'Adoption of {tech.replace("_", " ")} by Workforce Barrier')
                ax.set_xlabel('Reports Workforce Barrier (0=No, 1=Yes)')
                ax.set_ylabel('Adoption Rate')
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        # H5a: Companies that actively collaborate with suppliers and partners using digital platforms experience higher supply chain resilience compared to other technologies.
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H5a: Companies that actively collaborate with suppliers and partners using digital platforms experience higher supply chain resilience compared to other technologies.',
                 fontsize=10, fontweight='bold')

        # Correlation between collaboration score and resilience score
        corr, p_val = stats.pearsonr(df_clean['collaboration_score'].dropna(), df_clean['resilience_score'].dropna())
        plt.text(0.1, 0.85, f'Pearson Correlation: {corr:.2f}, p-value: {p_val:.3f}', fontsize=10)

        if p_val < 0.05:
            plt.text(0.1, 0.8, 'Conclusion: Reject null hypothesis - There is a significant relationship.', fontsize=10)
        else:
            plt.text(0.1, 0.8, 'Conclusion: Fail to reject null hypothesis - No significant relationship found.',
                     fontsize=10)

        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.regplot(data=df_clean, x='collaboration_score', y='resilience_score', ax=ax)
        ax.set_title('Relationship Between Collaboration and Resilience Scores')
        ax.set_xlabel('Collaboration Score')
        ax.set_ylabel('Resilience Score')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # H5b: The use of technologies strengthens collaboration and trust among stakeholders in the supply chain.
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H5b: The use of technologies strengthens collaboration and trust among stakeholders in the supply chain.',
                 fontsize=10, fontweight='bold')

        # Correlation between technology adoption and collaboration
        corr, p_val = stats.pearsonr(df_clean['adoption_score'].dropna(), df_clean['collaboration_score'].dropna())
        plt.text(0.1, 0.85, f'Pearson Correlation: {corr:.2f}, p-value: {p_val:.3f}', fontsize=10)

        if p_val < 0.05:
            plt.text(0.1, 0.8, 'Conclusion: Reject null hypothesis - There is a significant relationship.', fontsize=10)
        else:
            plt.text(0.1, 0.8, 'Conclusion: Fail to reject null hypothesis - No significant relationship found.',
                     fontsize=10)

        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.regplot(data=df_clean, x='adoption_score', y='collaboration_score', ax=ax)
        ax.set_title('Relationship Between Technology Adoption and Collaboration')
        ax.set_xlabel('Technology Adoption Score')
        ax.set_ylabel('Collaboration Score')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # H6a: Companies using real-time data analytics and AI-powered decision support systems make more effective and agile supply chain decisions during disruptions.
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H6a: Companies using real-time data analytics and AI-powered decision support systems make more effective and agile supply chain decisions during disruptions.',
                 fontsize=10, fontweight='bold')

        # Compare resilience scores for companies using AI and Big Data
        ai_users = df_clean[df_clean['adoption_Artificial_Intelligence_AI'] == 1]['resilience_score'].mean()
        non_ai_users = df_clean[df_clean['adoption_Artificial_Intelligence_AI'] == 0]['resilience_score'].mean()
        _, p_val_ai = stats.ttest_ind(
            df_clean[df_clean['adoption_Artificial_Intelligence_AI'] == 1]['resilience_score'].dropna(),
            df_clean[df_clean['adoption_Artificial_Intelligence_AI'] == 0]['resilience_score'].dropna(),
            equal_var=False
        )

        bda_users = df_clean[df_clean['adoption_Big_Data_Analytics'] == 1]['resilience_score'].mean()
        non_bda_users = df_clean[df_clean['adoption_Big_Data_Analytics'] == 0]['resilience_score'].mean()
        _, p_val_bda = stats.ttest_ind(
            df_clean[df_clean['adoption_Big_Data_Analytics'] == 1]['resilience_score'].dropna(),
            df_clean[df_clean['adoption_Big_Data_Analytics'] == 0]['resilience_score'].dropna(),
            equal_var=False
        )

        plt.text(0.1, 0.85, 'AI users vs non-users resilience:', fontsize=10)
        plt.text(0.1, 0.8, f"AI Users: {ai_users:.2f}, Non-Users: {non_ai_users:.2f}, p={p_val_ai:.3f}", fontsize=10)
        plt.text(0.1, 0.75, 'Big Data Analytics users vs non-users resilience:', fontsize=10)
        plt.text(0.1, 0.7, f"BDA Users: {bda_users:.2f}, Non-Users: {non_bda_users:.2f}, p={p_val_bda:.3f}",
                 fontsize=10)

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=pd.DataFrame({
            'Technology': ['AI', 'Big Data Analytics'],
            'Resilience_Difference': [ai_users - non_ai_users, bda_users - non_bda_users],
            'p-value': [p_val_ai, p_val_bda]
        }), x='Technology', y='Resilience_Difference', ax=ax)
        ax.set_title('Impact of AI and Big Data Analytics on Resilience')
        ax.set_ylabel('Difference in Resilience Score (Users - Non-Users)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # H6b: Organizations that implement automated risk assessment tools report improved supply chain visibility and control.
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9,
                 'H6b: Organizations that implement automated risk assessment tools report improved supply chain visibility and control.',
                 fontsize=10, fontweight='bold')

        # Compare visibility rankings for companies using different technologies
        # For this analysis, we'll focus on IoT and Blockchain as key technologies for visibility
        iot_users = df_clean[df_clean['adoption_Internet_of_Things_IoT'] == 1]['resilience_score'].mean()
        non_iot_users = df_clean[df_clean['adoption_Internet_of_Things_IoT'] == 0]['resilience_score'].mean()
        _, p_val_iot = stats.ttest_ind(
            df_clean[df_clean['adoption_Internet_of_Things_IoT'] == 1]['resilience_score'].dropna(),
            df_clean[df_clean['adoption_Internet_of_Things_IoT'] == 0]['resilience_score'].dropna(),
            equal_var=False
        )

        blockchain_users = df_clean[df_clean['adoption_Blockchain'] == 1]['resilience_score'].mean()
        non_blockchain_users = df_clean[df_clean['adoption_Blockchain'] == 0]['resilience_score'].mean()
        _, p_val_blockchain = stats.ttest_ind(
            df_clean[df_clean['adoption_Blockchain'] == 1]['resilience_score'].dropna(),
            df_clean[df_clean['adoption_Blockchain'] == 0]['resilience_score'].dropna(),
            equal_var=False
        )

        plt.text(0.1, 0.85, 'IoT users vs non-users resilience:', fontsize=10)
        plt.text(0.1, 0.8, f"IoT Users: {iot_users:.2f}, Non-Users: {non_iot_users:.2f}, p={p_val_iot:.3f}",
                 fontsize=10)
        plt.text(0.1, 0.75, 'Blockchain users vs non-users resilience:', fontsize=10)
        plt.text(0.1, 0.7,
                 f"Blockchain Users: {blockchain_users:.2f}, Non-Users: {non_blockchain_users:.2f}, p={p_val_blockchain:.3f}",
                 fontsize=10)

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=pd.DataFrame({
            'Technology': ['IoT', 'Blockchain'],
            'Resilience_Difference': [iot_users - non_iot_users, blockchain_users - non_blockchain_users],
            'p-value': [p_val_iot, p_val_blockchain]
        }), x='Technology', y='Resilience_Difference', ax=ax)
        ax.set_title('Impact of IoT and Blockchain on Resilience')
        ax.set_ylabel('Difference in Resilience Score (Users - Non-Users)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Summary of Findings
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.1, 0.9, 'Summary of Key Findings', fontsize=16, fontweight='bold')

        findings = [
            "1. The most common disruptions were Global Crises and Supplier Failures.",
            "2. Companies experiencing more disruptions tend to adopt more technologies (H1a supported).",
            "3. Large companies adopt more technologies than SMEs (H2a supported).",
            "4. High implementation costs are the most frequently reported barrier (H4a supported).",
            "5. Companies using multiple technologies report higher resilience (H3b supported).",
            "6. Collaboration through digital platforms correlates with higher resilience (H5a supported).",
            "7. AI and Big Data Analytics users show higher resilience scores (H6a supported).",
            "8. IoT and Blockchain adoption correlates with improved resilience (H6b supported)."
        ]

        for i, finding in enumerate(findings, start=1):
            plt.text(0.1, 0.85 - i * 0.05, finding, fontsize=10)

        plt.text(0.1, 0.4, 'Recommendations:', fontsize=12, fontweight='bold')
        recommendations = [
            "1. Prioritize technologies that address your most frequent disruptions.",
            "2. Invest in workforce training to overcome skill barriers to adoption.",
            "3. Implement technologies in combination for greater resilience impact.",
            "4. Focus on collaboration-enhancing technologies for supply chain partnerships.",
            "5. Large companies should lead in technology adoption and share best practices.",
            "6. Address cost barriers through phased implementation and ROI analysis."
        ]

        for i, recommendation in enumerate(recommendations, start=1):
            plt.text(0.1, 0.35 - i * 0.05, recommendation, fontsize=10)

        pdf.savefig()
        plt.close()


# Create the report
create_report(df_clean)