import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import textwrap
from fpdf import FPDF
import os
import traceback
from datetime import datetime

# Set up visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


class SupplyChainAnalyzer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.clean_df = None
        self.report_text = []
        self.current_section = ""

    def load_data(self):
        """Load the survey data from CSV"""
        self.df = pd.read_csv(self.data_file)
        self.report_text.append(("Data Loading",
                                 f"Successfully loaded data with {len(self.df)} records and {len(self.df.columns)} columns."))

    def initial_data_exploration(self):
        """Perform initial exploration of the data"""
        exploration = {
            "Missing Values": self.df.isnull().sum().sum(),
            "Duplicate Records": self.df.duplicated().sum(),
            "Data Types": self.df.dtypes.value_counts().to_dict(),
            "Numeric Columns": self.df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "Categorical Columns": self.df.select_dtypes(include=['object', 'bool']).columns.tolist()
        }

        self.report_text.append(("Initial Data Exploration", str(exploration)))

        # Display basic info
        print("Data Info:")
        print(self.df.info())

        print("\nSummary Statistics:")
        print(self.df.describe(include='all'))

    def clean_data(self):
        """Clean and preprocess the data"""
        # Create a copy for cleaning
        self.clean_df = self.df.copy()

        # Handle missing values for key columns
        key_columns = [
            'what_industry_does_your_company_belong_to',
            'what_is_your_role_in_the_company',
            'what_is_the_size_of_your_company',
            'disruption_count',
            'tech_adoption_count',
            'resilience_score'
        ]

        for col in key_columns:
            if col in self.clean_df.columns:
                if self.clean_df[col].dtype in ['int64', 'float64']:
                    self.clean_df[col] = self.clean_df[col].fillna(0)
                else:
                    self.clean_df[col] = self.clean_df[col].fillna('Unknown')

        # Clean categorical columns
        categorical_cols = self.clean_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.clean_df[col] = self.clean_df[col].str.strip()
            self.clean_df[col] = self.clean_df[col].replace(['', 'NA', 'N/A', 'n/a', 'na', 'NaN'], 'Unknown')

        # Convert boolean columns (assuming columns with True/False values)
        bool_cols = [
            'natural_disasters', 'supplier_failures', 'cybersecurity_threats',
            'global_crises', 'transportation_delays', 'high_tech_company'
        ]

        for col in bool_cols:
            if col in self.clean_df.columns:
                self.clean_df[col] = self.clean_df[col].astype(bool)

        # Clean industry categories
        if 'what_industry_does_your_company_belong_to' in self.clean_df.columns:
            self.clean_df['industry_group'] = self.clean_df['what_industry_does_your_company_belong_to'].apply(
                lambda x: 'manufacturing' if x in ['automotive', 'consumer goods', 'electronics'] else 'other'
            )

        # Clean company size
        if 'what_is_the_size_of_your_company' in self.clean_df.columns:
            size_mapping = {'small': 1, 'medium': 2, 'large': 3}
            self.clean_df['company_size_num'] = self.clean_df['what_is_the_size_of_your_company'].map(size_mapping)
            self.clean_df['company_size_num'] = self.clean_df['company_size_num'].fillna(
                2)  # Default to medium if missing

        # Calculate disruption count if not already present
        if 'disruption_count' not in self.clean_df.columns:
            disruption_cols = ['natural_disasters', 'supplier_failures', 'cybersecurity_threats',
                               'global_crises', 'transportation_delays']
            self.clean_df['disruption_count'] = self.clean_df[disruption_cols].sum(axis=1)

        # Calculate technology adoption count if not already present
        if 'tech_adoption_count' not in self.clean_df.columns:
            tech_cols = [
                'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_artificial_intelligence_ai',
                'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_internet_of_things_iot',
                'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_blockchain',
                'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_cloud_computing',
                'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_robotics_amp_automation',
                'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_big_data_analytics',
                'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_simulation'
            ]
            self.clean_df['tech_adoption_count'] = self.clean_df[tech_cols].sum(axis=1)

        # Handle resilience score if not present
        if 'resilience_score' not in self.clean_df.columns:
            # Create a simple resilience score based on technology implementation extent
            tech_implementation_cols = [
                'to_what_extent_has_your_company_implemented_the_following_technologies_likert_scale_1_not_implemented_5_fully_implemented_ai',
                'to_what_extent_has_your_company_implemented_the_following_technologies_likert_scale_1_not_implemented_5_fully_implemented_iot',
                'to_what_extent_has_your_company_implemented_the_following_technologies_likert_scale_1_not_implemented_5_fully_implemented_blockchain',
                'to_what_extent_has_your_company_implemented_the_following_technologies_likert_scale_1_not_implemented_5_fully_implemented_cloud_computing',
                'to_what_extent_has_your_company_implemented_the_following_technologies_likert_scale_1_not_implemented_5_fully_implemented_robotics_amp_automation',
                'to_what_extent_has_your_company_implemented_the_following_technologies_likert_scale_1_not_implemented_5_fully_implemented_big_data_analytics'
            ]

            # Calculate average implementation score
            implementation_scores = self.clean_df[tech_implementation_cols].mean(axis=1)
            self.clean_df['resilience_score'] = implementation_scores.fillna(0)

        self.report_text.append(("Data Cleaning",
                                 "Completed data cleaning including handling missing values, standardizing categories, and creating derived metrics."))

    def visualize_data_distribution(self):
        """Create visualizations for data distribution"""
        plt.figure(figsize=(15, 10))

        # 1. Industry Distribution
        plt.subplot(2, 2, 1)
        industry_counts = self.clean_df['what_industry_does_your_company_belong_to'].value_counts()
        sns.barplot(x=industry_counts.values, y=industry_counts.index, hue=industry_counts.index, palette='viridis',
                    legend=False)
        plt.title('Industry Distribution')
        plt.xlabel('Count')
        plt.ylabel('Industry')

        # 2. Company Size Distribution
        plt.subplot(2, 2, 2)
        size_counts = self.clean_df['what_is_the_size_of_your_company'].value_counts()
        sns.barplot(x=size_counts.values, y=size_counts.index, hue=size_counts.index, palette='viridis', legend=False)
        plt.title('Company Size Distribution')
        plt.xlabel('Count')
        plt.ylabel('Company Size')

        # 3. Disruption Count Distribution
        plt.subplot(2, 2, 3)
        sns.histplot(self.clean_df['disruption_count'], bins=6, kde=True)
        plt.title('Distribution of Disruption Counts')
        plt.xlabel('Number of Disruption Types Experienced')
        plt.ylabel('Count')

        # 4. Technology Adoption Count Distribution
        plt.subplot(2, 2, 4)
        sns.histplot(self.clean_df['tech_adoption_count'], bins=8, kde=True)
        plt.title('Distribution of Technology Adoption Counts')
        plt.xlabel('Number of Technologies Adopted')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig('data_distribution.png')
        plt.close()

        self.report_text.append(("Data Distribution Visualization",
                                 "Created visualizations showing industry distribution, company size, disruption counts, and technology adoption."))

    def test_hypothesis_h1(self):
        """Test Hypothesis H1: Disruptions and Supply Chain Resilience"""
        self.current_section = "H1: Disruptions and Supply Chain Resilience"

        # H1a: Companies that have experienced higher levels of supply chain disruptions are more likely to adopt resilience-enhancing technologies.
        # Correlation between disruption_count and tech_adoption_count
        valid_data = self.clean_df[['disruption_count', 'tech_adoption_count']].dropna()

        if len(valid_data) >= 3:
            corr, p_value = stats.pearsonr(
                valid_data['disruption_count'],
                valid_data['tech_adoption_count']
            )

            result_h1a = {
                "Hypothesis": "H1a: Higher disruptions lead to more technology adoption",
                "Correlation": corr,
                "P-value": p_value,
                "Sample Size": len(valid_data),
                "Conclusion": f"There is {'a significant' if p_value < 0.05 else 'no significant'} relationship between disruption count and technology adoption (r={corr:.2f}, p={p_value:.4f})."
            }
        else:
            result_h1a = {
                "Hypothesis": "H1a: Higher disruptions lead to more technology adoption",
                "Conclusion": "Insufficient data to calculate correlation (need at least 3 complete observations)"
            }

        self.report_text.append((self.current_section, result_h1a))

        # Visualization for H1a if we have data
        if len(valid_data) > 0:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x='disruption_count',
                y='tech_adoption_count',
                hue=self.clean_df['what_is_the_size_of_your_company'],
                data=valid_data,
                palette='viridis'
            )
            plt.title('Technology Adoption vs. Disruption Count')
            plt.xlabel('Number of Disruption Types Experienced')
            plt.ylabel('Number of Technologies Adopted')
            plt.legend(title='Company Size')
            plt.savefig('h1a_disruption_vs_tech_adoption.png')
            plt.close()

        # H1b: The impact of disruptions significantly influences the choice of resilience-enhancing technologies.
        disruption_cols = ['natural_disasters', 'supplier_failures', 'cybersecurity_threats',
                           'global_crises', 'transportation_delays']
        tech_cols = [
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_artificial_intelligence_ai',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_internet_of_things_iot',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_blockchain',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_cloud_computing',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_robotics_amp_automation',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_big_data_analytics',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_simulation'
        ]
        tech_names = ['AI', 'IoT', 'Blockchain', 'Cloud Computing', 'Robotics & Automation', 'Big Data Analytics',
                      'Simulation']

        results_h1b = []

        for i, disruption in enumerate(disruption_cols):
            for j, tech in enumerate(tech_cols):
                try:
                    contingency = pd.crosstab(self.clean_df[disruption], self.clean_df[tech])

                    # Only perform chi-square if we have sufficient data
                    if contingency.size >= 4 and contingency.values.min() >= 5:
                        chi2, p, dof, expected = chi2_contingency(contingency)

                        if p < 0.05:
                            results_h1b.append({
                                "Disruption": disruption,
                                "Technology": tech_names[j],
                                "Chi-square": chi2,
                                "P-value": p,
                                "Conclusion": f"Specific disruption type ({disruption}) is associated with adoption of {tech_names[j]}"
                            })
                except Exception as e:
                    results_h1b.append({
                        "Disruption": disruption,
                        "Technology": tech_names[j],
                        "Error": str(e)
                    })

        if not results_h1b:
            results_h1b.append(
                "No significant associations found between specific disruption types and technology choices.")

        self.report_text.append((self.current_section + " - H1b", results_h1b))

        # Visualization for H1b - Heatmap of disruption types vs technology adoption
        tech_adoption_by_disruption = pd.DataFrame()

        for disruption in disruption_cols:
            for tech, tech_name in zip(tech_cols, tech_names):
                adoption_rate = self.clean_df.groupby(disruption)[tech].mean()
                if True in adoption_rate.index:
                    tech_adoption_by_disruption.loc[disruption, tech_name] = adoption_rate[True]
                else:
                    tech_adoption_by_disruption.loc[disruption, tech_name] = 0

        if not tech_adoption_by_disruption.empty:
            plt.figure(figsize=(12, 6))
            sns.heatmap(tech_adoption_by_disruption, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Technology Adoption Rates by Disruption Type')
            plt.xlabel('Technology')
            plt.ylabel('Disruption Type')
            plt.savefig('h1b_disruption_tech_heatmap.png')
            plt.close()

    def test_hypothesis_h2(self):
        """Test Hypothesis H2: Adoption of Technologies"""
        self.current_section = "H2: Adoption of Technologies"

        # H2a: The adoption rate of AI, IoT, and Big Data Analytics is higher in large enterprises compared to SMEs
        tech_cols = [
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_artificial_intelligence_ai',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_internet_of_things_iot',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_big_data_analytics'
        ]
        tech_names = ['AI', 'IoT', 'Big Data Analytics']

        results_h2a = []

        # Create a simplified company size column (SME vs Large)
        self.clean_df['company_size_category'] = self.clean_df['what_is_the_size_of_your_company'].apply(
            lambda x: 'SME' if x in ['small', 'medium'] else 'Large'
        )

        for tech_col, tech_name in zip(tech_cols, tech_names):
            try:
                # Create contingency table
                contingency = pd.crosstab(self.clean_df['company_size_category'], self.clean_df[tech_col])

                # Perform chi-square test only if we have sufficient data
                if contingency.size >= 4 and contingency.values.min() >= 5:
                    chi2, p, dof, expected = chi2_contingency(contingency)

                    results_h2a.append({
                        "Technology": tech_name,
                        "Chi-square": chi2,
                        "P-value": p,
                        "Conclusion": f"Adoption of {tech_name} {'differs' if p < 0.05 else 'does not differ'} between SMEs and Large enterprises"
                    })

                    # Calculate adoption rates
                    adoption_rates = self.clean_df.groupby('company_size_category')[tech_col].mean()
                    results_h2a[-1]['Adoption Rate - SME'] = adoption_rates['SME']
                    results_h2a[-1]['Adoption Rate - Large'] = adoption_rates['Large']
                else:
                    results_h2a.append({
                        "Technology": tech_name,
                        "Conclusion": "Insufficient data for chi-square test"
                    })
            except Exception as e:
                results_h2a.append({
                    "Technology": tech_name,
                    "Error": str(e)
                })

        self.report_text.append((self.current_section + " - H2a", results_h2a))

        # Visualization for H2a
        adoption_rates = pd.DataFrame()
        for tech_col, tech_name in zip(tech_cols, tech_names):
            if tech_col in self.clean_df.columns:
                rates = self.clean_df.groupby('company_size_category')[tech_col].mean()
                adoption_rates[tech_name] = rates

        if not adoption_rates.empty:
            adoption_rates = adoption_rates.T
            adoption_rates.plot(kind='bar', figsize=(10, 6))
            plt.title('Technology Adoption Rates by Company Size')
            plt.ylabel('Adoption Rate')
            plt.xlabel('Technology')
            plt.xticks(rotation=45)
            plt.legend(title='Company Size')
            plt.tight_layout()
            plt.savefig('h2a_tech_adoption_by_size.png')
            plt.close()

        # H2b: Companies in highly disrupted industries (e.g., automotive, electronics) are more likely to implement Cloud Computing and Blockchain for resilience.
        # Define highly disrupted industries
        self.clean_df['high_disruption_industry'] = self.clean_df['what_industry_does_your_company_belong_to'].isin(
            ['automotive', 'electronics', 'consumer goods']
        )

        tech_cols_h2b = [
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_cloud_computing',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_blockchain'
        ]
        tech_names_h2b = ['Cloud Computing', 'Blockchain']

        results_h2b = []

        for tech_col, tech_name in zip(tech_cols_h2b, tech_names_h2b):
            try:
                # Create contingency table
                contingency = pd.crosstab(self.clean_df['high_disruption_industry'], self.clean_df[tech_col])

                # Perform chi-square test only if we have sufficient data
                if contingency.size >= 4 and contingency.values.min() >= 5:
                    chi2, p, dof, expected = chi2_contingency(contingency)

                    # Calculate adoption rates
                    adoption_rates = self.clean_df.groupby('high_disruption_industry')[tech_col].mean()

                    results_h2b.append({
                        "Technology": tech_name,
                        "Chi-square": chi2,
                        "P-value": p,
                        "Adoption Rate - High Disruption Industries": adoption_rates[True],
                        "Adoption Rate - Other Industries": adoption_rates[False],
                        "Conclusion": f"Adoption of {tech_name} {'differs' if p < 0.05 else 'does not differ'} between high-disruption and other industries"
                    })
                else:
                    results_h2b.append({
                        "Technology": tech_name,
                        "Conclusion": "Insufficient data for chi-square test"
                    })
            except Exception as e:
                results_h2b.append({
                    "Technology": tech_name,
                    "Error": str(e)
                })

        self.report_text.append((self.current_section + " - H2b", results_h2b))

        # Visualization for H2b
        adoption_rates_h2b = pd.DataFrame()
        for tech_col, tech_name in zip(tech_cols_h2b, tech_names_h2b):
            if tech_col in self.clean_df.columns:
                rates = self.clean_df.groupby('high_disruption_industry')[tech_col].mean()
                adoption_rates_h2b[tech_name] = rates

        if not adoption_rates_h2b.empty:
            adoption_rates_h2b = adoption_rates_h2b.T
            adoption_rates_h2b.columns = ['Other Industries', 'High Disruption Industries']
            adoption_rates_h2b.plot(kind='bar', figsize=(10, 6))
            plt.title('Technology Adoption Rates by Industry Disruption Level')
            plt.ylabel('Adoption Rate')
            plt.xlabel('Technology')
            plt.xticks(rotation=45)
            plt.legend(title='Industry Type')
            plt.tight_layout()
            plt.savefig('h2b_tech_adoption_by_industry.png')
            plt.close()

    def test_hypothesis_h3(self):
        """Test Hypothesis H3: Effectiveness of Technologies"""
        self.current_section = "H3: Effectiveness of Technologies"

        # H3a: Companies using AI and IoT report higher supply chain efficiency and resilience compared to those not using them.
        tech_cols = [
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_artificial_intelligence_ai',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_internet_of_things_iot'
        ]
        tech_names = ['AI', 'IoT']

        results_h3a = []

        for tech_col, tech_name in zip(tech_cols, tech_names):
            if tech_col in self.clean_df.columns:
                # Split data into users and non-users
                users = self.clean_df[self.clean_df[tech_col] == True]['resilience_score'].dropna()
                non_users = self.clean_df[self.clean_df[tech_col] == False]['resilience_score'].dropna()

                # Perform t-test only if we have data in both groups
                if len(users) > 1 and len(non_users) > 1:
                    t_stat, p_value = ttest_ind(users, non_users, equal_var=False)

                    results_h3a.append({
                        "Technology": tech_name,
                        "Mean Resilience Score - Users": users.mean(),
                        "Mean Resilience Score - Non-users": non_users.mean(),
                        "T-statistic": t_stat,
                        "P-value": p_value,
                        "Conclusion": f"Companies using {tech_name} {'have' if p_value < 0.05 else 'do not have'} significantly different resilience scores"
                    })
                else:
                    results_h3a.append({
                        "Technology": tech_name,
                        "Conclusion": f"Insufficient data to compare {tech_name} users and non-users"
                    })

        self.report_text.append((self.current_section + " - H3a", results_h3a))

        # Visualization for H3a
        for tech_col, tech_name in zip(tech_cols, tech_names):
            if tech_col in self.clean_df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    x=tech_col,
                    y='resilience_score',
                    hue=tech_col,  # Add hue to address warning
                    data=self.clean_df,
                    palette='viridis',
                    legend=False
                )
                plt.title(f'Resilience Score by {tech_name} Adoption')
                plt.xlabel(f'Uses {tech_name}')
                plt.ylabel('Resilience Score')
                plt.xticks([0, 1], ['No', 'Yes'])
                plt.savefig(f'h3a_resilience_{tech_name.lower()}.png')
                plt.close()

        # H3b: Companies using multiple technologies in combination report better resilience scores than those using a single technology.
        tech_cols_h3b = [
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_artificial_intelligence_ai',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_internet_of_things_iot',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_blockchain',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_cloud_computing'
        ]

        if all(col in self.clean_df.columns for col in tech_cols_h3b):
            self.clean_df['tech_combination'] = self.clean_df[tech_cols_h3b].sum(axis=1)

            # Group by number of technologies used
            grouped = self.clean_df.groupby('tech_combination')['resilience_score'].agg(['mean', 'count'])
            grouped = grouped[grouped['count'] > 5]  # Filter groups with sufficient data

            # Only proceed with ANOVA if we have at least two groups
            if len(grouped) >= 2:
                groups = []
                for num_tech in grouped.index:
                    groups.append(
                        self.clean_df[self.clean_df['tech_combination'] == num_tech]['resilience_score'].dropna())

                try:
                    f_stat, p_value = f_oneway(*groups)

                    results_h3b = {
                        "F-statistic": f_stat,
                        "P-value": p_value,
                        "Mean Resilience by Tech Count": grouped['mean'].to_dict(),
                        "Conclusion": f"Resilience scores {'differ' if p_value < 0.05 else 'do not differ'} significantly based on number of technologies adopted"
                    }
                except Exception as e:
                    results_h3b = {
                        "Error": str(e),
                        "Conclusion": "Could not perform ANOVA due to error"
                    }
            else:
                results_h3b = {
                    "Conclusion": "Insufficient data to perform ANOVA (need at least two groups with sufficient data)"
                }

            self.report_text.append((self.current_section + " - H3b", results_h3b))

            # Visualization for H3b if we have data
            if len(grouped) > 0:
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    x='tech_combination',
                    y='resilience_score',
                    hue='tech_combination',  # Add hue to address warning
                    data=self.clean_df[self.clean_df['tech_combination'].isin(grouped.index)],
                    palette='viridis',
                    legend=False
                )
                plt.title('Resilience Score by Number of Technologies Adopted')
                plt.xlabel('Number of Technologies Used')
                plt.ylabel('Resilience Score')
                plt.savefig('h3b_resilience_by_tech_count.png')
                plt.close()

    def test_hypothesis_h4(self):
        """Test Hypothesis H4: Barriers to Adoption"""
        self.current_section = "H4: Barriers to Adoption"

        # H4a: High implementation costs are the primary barrier to adopting new supply chain technologies.
        barrier_cols = [
            'what_were_the_main_challenges_in_implementing_these_technologies_select_all_that_apply_high_implementation_costs',
            'what_were_the_main_challenges_in_implementing_these_technologies_select_all_that_apply_lack_of_skilled_workforce',
            'what_were_the_main_challenges_in_implementing_these_technologies_select_all_that_apply_integration_issues',
            'what_were_the_main_challenges_in_implementing_these_technologies_select_all_that_apply_resistance_to_change'
        ]
        barrier_names = ['High Costs', 'Lack of Skilled Workforce', 'Integration Issues', 'Resistance to Change']

        if all(col in self.clean_df.columns for col in barrier_cols):
            # Convert barrier columns to boolean (handle mixed types)
            barrier_data = self.clean_df[barrier_cols].copy()

            # Convert to boolean - handle different possible representations
            for col in barrier_cols:
                if barrier_data[col].dtype == 'object':
                    # Convert string representations to boolean
                    barrier_data[col] = barrier_data[col].apply(
                        lambda x: True if str(x).lower() in ['true', 'yes', '1'] else False
                    )

            # Now we can safely sum the boolean values
            barrier_counts = barrier_data.sum().sort_values(ascending=False)

            results_h4a = {
                "Barrier Prevalence": barrier_counts.to_dict(),
                "Primary Barrier": barrier_counts.idxmax(),
                "Conclusion": f"The most common barrier is {barrier_counts.idxmax()}"
            }

            self.report_text.append((self.current_section + " - H4a", results_h4a))

            # Visualization for H4a
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=barrier_counts.values,
                y=barrier_counts.index,
                hue=barrier_counts.index,  # Add hue to address warning
                palette='viridis',
                legend=False
            )
            plt.title('Prevalence of Implementation Barriers')
            plt.xlabel('Number of Companies Reporting Barrier')
            plt.ylabel('Barrier Type')
            plt.savefig('h4a_barrier_prevalence.png')
            plt.close()

        # H4b: Lack of skilled workforce significantly affects the adoption of AI and Big Data Analytics in supply chain management.
        tech_cols_h4b = [
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_artificial_intelligence_ai',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_big_data_analytics'
        ]
        tech_names_h4b = ['AI', 'Big Data Analytics']

        skill_barrier_col = 'what_were_the_main_challenges_in_implementing_these_technologies_select_all_that_apply_lack_of_skilled_workforce'

        if (skill_barrier_col in self.clean_df.columns and
                all(col in self.clean_df.columns for col in tech_cols_h4b)):

            # Convert skill barrier column to boolean
            self.clean_df['skill_barrier_bool'] = self.clean_df[skill_barrier_col].apply(
                lambda x: True if str(x).lower() in ['true', 'yes', '1'] else False
            )

            results_h4b = []

            for tech_col, tech_name in zip(tech_cols_h4b, tech_names_h4b):
                try:
                    # Create contingency table
                    contingency = pd.crosstab(
                        self.clean_df['skill_barrier_bool'],
                        self.clean_df[tech_col]
                    )

                    # Perform chi-square test only if we have sufficient data
                    if contingency.shape == (2, 2) and contingency.values.min() >= 5:
                        chi2, p, dof, expected = chi2_contingency(contingency)

                        results_h4b.append({
                            "Technology": tech_name,
                            "Chi-square": chi2,
                            "P-value": p,
                            "Conclusion": f"Lack of skilled workforce {'is' if p < 0.05 else 'is not'} significantly associated with adoption of {tech_name}"
                        })
                    else:
                        results_h4b.append({
                            "Technology": tech_name,
                            "Conclusion": "Insufficient data for chi-square test"
                        })
                except Exception as e:
                    results_h4b.append({
                        "Technology": tech_name,
                        "Error": str(e)
                    })

            self.report_text.append((self.current_section + " - H4b", results_h4b))

            # Visualization for H4b
            adoption_by_skill_barrier = pd.DataFrame()
            for tech_col, tech_name in zip(tech_cols_h4b, tech_names_h4b):
                # Group by skill barrier and calculate mean adoption
                grouped = self.clean_df.groupby('skill_barrier_bool')[tech_col].mean()
                adoption_by_skill_barrier[tech_name] = grouped

            if not adoption_by_skill_barrier.empty:
                adoption_by_skill_barrier.plot(kind='bar', figsize=(10, 6))
                plt.title('Technology Adoption by Skill Barrier Presence')
                plt.ylabel('Adoption Rate')
                plt.xlabel('Experienced Skill Barrier')
                plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
                plt.legend(title='Technology')
                plt.tight_layout()
                plt.savefig('h4b_adoption_by_skill_barrier.png')
                plt.close()

    def test_hypothesis_h5(self):
        """Test Hypothesis H5: Collaboration and Stakeholder Engagement"""
        self.current_section = "H5: Collaboration and Stakeholder Engagement"

        # H5a: Companies that actively collaborate with suppliers and partners using digital platforms experience higher supply chain resilience.
        if (
                'to_what_extent_have_new_technologies_improved_communication_and_coordination_with_your_suppliers_and_partners_1_not_at_all_2_slightly_3_moderately_4_highly_5_very_high' in self.clean_df.columns and
                'resilience_score' in self.clean_df.columns):

            # Rename for easier access
            self.clean_df['collaboration_score'] = self.clean_df[
                'to_what_extent_have_new_technologies_improved_communication_and_coordination_with_your_suppliers_and_partners_1_not_at_all_2_slightly_3_moderately_4_highly_5_very_high']

            # Get only rows where both scores are available
            valid_data = self.clean_df[['collaboration_score', 'resilience_score']].dropna()

            if len(valid_data) >= 3:  # Minimum for correlation
                try:
                    # Correlation between collaboration score and resilience score
                    corr, p_value = stats.pearsonr(
                        valid_data['collaboration_score'],
                        valid_data['resilience_score']
                    )

                    results_h5a = {
                        "Correlation": corr,
                        "P-value": p_value,
                        "Sample Size": len(valid_data),
                        "Conclusion": f"There is {'a significant' if p_value < 0.05 else 'no significant'} relationship between collaboration and resilience (r={corr:.2f}, p={p_value:.4f})."
                    }
                except Exception as e:
                    results_h5a = {
                        "Error": str(e),
                        "Conclusion": "Could not calculate correlation due to error"
                    }
            else:
                results_h5a = {
                    "Conclusion": "Insufficient data to calculate correlation (need at least 3 complete observations)"
                }

            self.report_text.append((self.current_section + " - H5a", results_h5a))

            # Visualization for H5a if we have data
            if len(valid_data) > 0:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(
                    x='collaboration_score',
                    y='resilience_score',
                    hue=self.clean_df['what_is_the_size_of_your_company'],
                    data=valid_data,
                    palette='viridis'
                )
                plt.title('Resilience Score vs. Collaboration Score')
                plt.xlabel('Collaboration Score (1-5)')
                plt.ylabel('Resilience Score')
                plt.legend(title='Company Size')
                plt.savefig('h5a_collaboration_vs_resilience.png')
                plt.close()

        # H5b: The use of technology strengthens collaboration and trust among stakeholders in the supply chain.
        if ('collaboration_score' in self.clean_df.columns and
                'tech_adoption_count' in self.clean_df.columns):

            # Get only rows where both variables are available
            valid_data = self.clean_df[['tech_adoption_count', 'collaboration_score']].dropna()

            if len(valid_data) >= 3:
                try:
                    corr, p_value = stats.pearsonr(
                        valid_data['tech_adoption_count'],
                        valid_data['collaboration_score']
                    )

                    results_h5b = {
                        "Correlation": corr,
                        "P-value": p_value,
                        "Sample Size": len(valid_data),
                        "Conclusion": f"There is {'a significant' if p_value < 0.05 else 'no significant'} relationship between technology adoption and collaboration (r={corr:.2f}, p={p_value:.4f})."
                    }
                except Exception as e:
                    results_h5b = {
                        "Error": str(e),
                        "Conclusion": "Could not calculate correlation due to error"
                    }
            else:
                results_h5b = {
                    "Conclusion": "Insufficient data to calculate correlation (need at least 3 complete observations)"
                }

            self.report_text.append((self.current_section + " - H5b", results_h5b))

            # Visualization for H5b if we have data
            if len(valid_data) > 0:
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    x='tech_adoption_count',
                    y='collaboration_score',
                    hue='tech_adoption_count',  # Add hue to address warning
                    data=valid_data,
                    palette='viridis',
                    legend=False
                )
                plt.title('Collaboration Score by Technology Adoption Count')
                plt.xlabel('Number of Technologies Adopted')
                plt.ylabel('Collaboration Score (1-5)')
                plt.savefig('h5b_collaboration_by_tech_adoption.png')
                plt.close()

    def test_hypothesis_h6(self):
        """Test Hypothesis H6: Control Mechanisms and Decision-Making"""
        self.current_section = "H6: Control Mechanisms and Decision-Making"

        # H6a: Companies using real-time data analytics and AI-powered decision support systems make more effective and agile supply chain decisions during disruptions.
        tech_cols_h6a = [
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_artificial_intelligence_ai',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_big_data_analytics'
        ]

        if all(col in self.clean_df.columns for col in tech_cols_h6a):
            # Create a combined AI/Analytics user group
            self.clean_df['ai_analytics_user'] = self.clean_df[tech_cols_h6a].any(axis=1)

            # Compare resilience scores
            users = self.clean_df[self.clean_df['ai_analytics_user'] == True]['resilience_score'].dropna()
            non_users = self.clean_df[self.clean_df['ai_analytics_user'] == False]['resilience_score'].dropna()

            # Only perform t-test if we have sufficient data
            if len(users) > 1 and len(non_users) > 1:
                try:
                    t_stat, p_value = ttest_ind(users, non_users, equal_var=False)

                    results_h6a = {
                        "Mean Resilience - AI/Analytics Users": users.mean(),
                        "Mean Resilience - Non-users": non_users.mean(),
                        "T-statistic": t_stat,
                        "P-value": p_value,
                        "Conclusion": f"Companies using AI/Analytics {'have' if p_value < 0.05 else 'do not have'} significantly different resilience scores"
                    }
                except Exception as e:
                    results_h6a = {
                        "Error": str(e),
                        "Conclusion": "Could not perform t-test due to error"
                    }
            else:
                results_h6a = {
                    "Conclusion": "Insufficient data to compare AI/Analytics users and non-users"
                }

            self.report_text.append((self.current_section + " - H6a", results_h6a))

            # Visualization for H6a if we have data
            if len(users) > 0 or len(non_users) > 0:
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    x='ai_analytics_user',
                    y='resilience_score',
                    hue='ai_analytics_user',  # Add hue to address warning
                    data=self.clean_df,
                    palette='viridis',
                    legend=False
                )
                plt.title('Resilience Score by AI/Analytics Adoption')
                plt.xlabel('Uses AI or Big Data Analytics')
                plt.ylabel('Resilience Score')
                plt.xticks([0, 1], ['No', 'Yes'])
                plt.savefig('h6a_resilience_by_ai_analytics.png')
                plt.close()

        # H6b: Organizations that implement automated risk assessment tools report improved supply chain visibility and control.
        tech_cols_h6b = [
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_internet_of_things_iot',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_blockchain'
        ]

        if all(col in self.clean_df.columns for col in tech_cols_h6b):
            # Create a combined IoT/Blockchain user group
            self.clean_df['visibility_tools_user'] = self.clean_df[tech_cols_h6b].any(axis=1)

            # Compare resilience scores
            users = self.clean_df[self.clean_df['visibility_tools_user'] == True]['resilience_score'].dropna()
            non_users = self.clean_df[self.clean_df['visibility_tools_user'] == False]['resilience_score'].dropna()

            # Only perform t-test if we have sufficient data
            if len(users) > 1 and len(non_users) > 1:
                try:
                    t_stat, p_value = ttest_ind(users, non_users, equal_var=False)

                    results_h6b = {
                        "Mean Resilience - IoT/Blockchain Users": users.mean(),
                        "Mean Resilience - Non-users": non_users.mean(),
                        "T-statistic": t_stat,
                        "P-value": p_value,
                        "Conclusion": f"Companies using IoT/Blockchain {'have' if p_value < 0.05 else 'do not have'} significantly different resilience scores"
                    }
                except Exception as e:
                    results_h6b = {
                        "Error": str(e),
                        "Conclusion": "Could not perform t-test due to error"
                    }
            else:
                results_h6b = {
                    "Conclusion": "Insufficient data to compare IoT/Blockchain users and non-users"
                }

            self.report_text.append((self.current_section + " - H6b", results_h6b))

            # Visualization for H6b if we have data
            if len(users) > 0 or len(non_users) > 0:
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    x='visibility_tools_user',
                    y='resilience_score',
                    hue='visibility_tools_user',  # Add hue to address warning
                    data=self.clean_df,
                    palette='viridis',
                    legend=False
                )
                plt.title('Resilience Score by IoT/Blockchain Adoption')
                plt.xlabel('Uses IoT or Blockchain')
                plt.ylabel('Resilience Score')
                plt.xticks([0, 1], ['No', 'Yes'])
                plt.savefig('h6b_resilience_by_iot_blockchain.png')
                plt.close()

    def analyze_qualitative_data(self):
        """Analyze qualitative responses from the survey"""
        self.current_section = "Qualitative Analysis"

        qualitative_cols = [
            'do_you_have_any_further_insights_on_how_technology_can_improve_supply_chain_resilience',
            'if_you_selected_other_in_the_previous_question_please_specify_the_technology_otherwise_you_may_leave_this_blank',
            'what_were_the_main_challenges_in_implementing_these_technologies_select_all_that_apply_other',
            'what_are_the_biggest_barriers_preventing_further_adoption_of_these_technologies_select_all_that_apply_other'
        ]

        results = {}

        for col in qualitative_cols:
            if col in self.clean_df.columns:
                # Get non-empty responses
                responses = self.clean_df[col].dropna()
                responses = responses[responses != 'Unknown']
                responses = responses[responses != '']
                responses = responses.astype(str)
                responses = responses[responses.str.strip() != '']

                if not responses.empty:
                    all_text = ' '.join(responses)

                    # Only proceed if we have at least 3 words
                    if len(all_text.split()) >= 3:
                        try:
                            wordcloud = WordCloud(width=800, height=400,
                                                background_color='white').generate(all_text)
                            plt.figure(figsize=(12, 6))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            plt.title(f'Word Cloud for {col[:30]}...')  # Truncate long titles
                            plt.savefig(f'wordcloud_{col[:20]}.png', bbox_inches='tight')
                            plt.close()

                            results[col] = {
                                "sample_responses": responses.head(5).tolist(),
                                "total_responses": len(responses),
                                "wordcloud_generated": True
                            }
                        except Exception as e:
                            results[col] = {
                                "sample_responses": responses.head(5).tolist(),
                                "total_responses": len(responses),
                                "wordcloud_generated": False,
                                "reason": str(e)
                            }
                    else:
                        results[col] = {
                            "note": "Insufficient text for word cloud",
                            "sample_responses": responses.head(5).tolist(),
                            "total_responses": len(responses)
                        }
                else:
                    results[col] = {
                        "note": "No valid responses found",
                        "total_responses": 0
                    }

        self.report_text.append((self.current_section, results))

    def generate_report(self):
        """Generate a PDF report with all findings"""

        class PDF(FPDF):
            def __init__(self):
                super().__init__()
                self.set_auto_page_break(auto=True, margin=15)
                self.set_margins(10, 10, 10)

            def header(self):
                self.set_font('helvetica', 'B', 12)
                self.cell(0, 10, 'Supply Chain Resilience Report',
                          new_x="LMARGIN", new_y="NEXT", align='C')
                self.ln(5)

            # Rest of the methods using 'helvetica' instead of 'DejaVu'

            def footer(self):
                self.set_y(-15)
                self.set_font('helvetica', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', align='C')

            def chapter_title(self, title):
                self.set_font('helvetica', 'B', 12)
                self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
                self.ln(4)

            def chapter_body(self, body):
                self.set_font('helvetica', '', 10)
                # Properly handle text wrapping
                for line in body.split('\n'):
                    if line.strip():  # Only process non-empty lines
                        self.multi_cell(0, 5, line.strip(), new_x="LMARGIN", new_y="NEXT")
                self.ln()

            def add_image(self, image_path, width=180):
                try:
                    if os.path.exists(image_path):
                        self.image(image_path, x=None, y=None, w=width)
                        self.ln(5)
                except Exception as e:
                    print(f"Error adding image {image_path}: {str(e)}")

        # Create PDF
        pdf = PDF()
        pdf.add_page()

        # Title page
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(0, 10, 'Supply Chain Resilience Technology Adoption Analysis',
                 new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.set_font('helvetica', '', 12)
        pdf.cell(0, 10, f'Report generated on {datetime.now().strftime("%Y-%m-%d")}',
                 new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(20)

        # Add introduction
        pdf.chapter_title('Introduction')
        intro_text = """
        This report presents findings from a survey assessing the impact of emerging technologies 
        on supply chain resilience. The analysis covers technology adoption patterns, effectiveness 
        in mitigating disruptions, barriers to implementation, and the role of collaboration in 
        enhancing supply chain resilience.
        """
        pdf.chapter_body(intro_text)

        # Add methodology
        pdf.chapter_title('Methodology')
        method_text = """
        The analysis included both quantitative and qualitative methods:
        - Hypothesis testing (t-tests, chi-square, ANOVA, correlation)
        - Descriptive statistics and visualizations
        - Qualitative analysis of open-ended responses
        - Examination of relationships between technology adoption and resilience metrics
        """
        pdf.chapter_body(method_text)

        # Add key findings section
        pdf.chapter_title('Key Findings')

        # Add content from each section
        for section, content in self.report_text:
            pdf.chapter_title(section)

            if isinstance(content, dict):
                # Format dictionary content
                body = "\n".join([f"{key}: {value}" for key, value in content.items()])
                pdf.chapter_body(body)

                # Add associated images if they exist
                image_files = {
                    'H1: Disruptions and Supply Chain Resilience': ['h1a_disruption_vs_tech_adoption.png',
                                                                    'h1b_disruption_tech_heatmap.png'],
                    'H2: Adoption of Technologies': ['h2a_tech_adoption_by_size.png',
                                                     'h2b_tech_adoption_by_industry.png'],
                    'H3: Effectiveness of Technologies': ['h3a_resilience_ai.png', 'h3a_resilience_iot.png',
                                                          'h3b_resilience_by_tech_count.png'],
                    'H4: Barriers to Adoption': ['h4a_barrier_prevalence.png', 'h4b_adoption_by_skill_barrier.png'],
                    'H5: Collaboration and Stakeholder Engagement': ['h5a_collaboration_vs_resilience.png',
                                                                     'h5b_collaboration_by_tech_adoption.png'],
                    'H6: Control Mechanisms and Decision-Making': ['h6a_resilience_by_ai_analytics.png',
                                                                   'h6b_resilience_by_iot_blockchain.png'],
                    'Data Distribution Visualization': ['data_distribution.png'],
                    'Qualitative Analysis': [f for f in os.listdir() if f.startswith('wordcloud_')]
                }

                if section in image_files:
                    for img_file in image_files[section]:
                        if os.path.exists(img_file):
                            try:
                                pdf.add_image(img_file)
                            except:
                                print(f"Could not add image {img_file}")

            elif isinstance(content, list):
                # Format list content
                if all(isinstance(item, dict) for item in content):
                    for item in content:
                        body = "\n".join([f"{key}: {value}" for key, value in item.items()])
                        pdf.chapter_body(body)
                        pdf.ln(3)
                else:
                    pdf.chapter_body("\n".join([str(item) for item in content]))
            else:
                pdf.chapter_body(str(content))

        # Add conclusions and recommendations
        pdf.chapter_title('Conclusions and Recommendations')
        conclusion_text = """
        Key Recommendations:
        1. Prioritize AI and IoT adoption for companies facing frequent disruptions
        2. Invest in workforce training to overcome skill barriers
        3. Implement technology combinations (AI + Blockchain + Cloud) for maximum resilience
        4. Focus on collaboration technologies to enhance supply chain visibility
        5. Address cost barriers through phased implementation plans

        Future Research Directions:
        - Longitudinal study on technology adoption impact
        - Industry-specific technology effectiveness
        - Ethical implications of AI in supply chain decisions
        """
        pdf.chapter_body(conclusion_text)

        # Save the PDF
        pdf_file = 'supply_chain_resilience_report.pdf'
        pdf.output(pdf_file)

        print(f"Report generated successfully: {pdf_file}")

    def run_full_analysis(self):
        """Run the complete analysis workflow"""
        print("Starting analysis...")
        self.load_data()
        self.initial_data_exploration()
        self.clean_data()
        self.visualize_data_distribution()

        # Test hypotheses
        self.test_hypothesis_h1()
        self.test_hypothesis_h2()
        self.test_hypothesis_h3()
        self.test_hypothesis_h4()
        self.test_hypothesis_h5()
        self.test_hypothesis_h6()

        # Analyze qualitative data
        self.analyze_qualitative_data()

        # Generate final report
        self.generate_report()
        print("Analysis completed successfully.")


# Main execution
if __name__ == "__main__":
    try:
        analyzer = SupplyChainAnalyzer("supply_chain_survey_data.csv")
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        traceback.print_exc()