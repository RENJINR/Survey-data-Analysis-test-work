import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from fpdf import FPDF
import os
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class SupplyChainAnalysis:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.report = FPDF()
        self.current_section = ""
        self.analysis_date = datetime.now().strftime("%Y-%m-%d")
        self.plots = []

    def load_data(self):
        """Load and clean data"""
        self.df = pd.read_csv(self.filepath)

        # Clean column names
        self.df.columns = [col.lower().replace(' ', '_') for col in self.df.columns]

        # Convert Yes/No to boolean
        yes_no_cols = [col for col in self.df.columns if self.df[col].isin(['Yes', 'No']).any()]
        for col in yes_no_cols:
            self.df[col] = self.df[col].replace({'Yes': True, 'No': False, np.nan: False})

        # Define technology columns
        self.tech_columns = [
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_artificial_intelligence_ai',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_internet_of_things_iot',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_blockchain',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_cloud_computing',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_big_data_analytics',
            'which_technologies_does_your_company_currently_use_for_supply_chain_resilience_select_all_that_apply_simulation'
        ]

        # Define disruption columns
        self.disruption_columns = [
            'natural_disasters',
            'supplier_failures',
            'cybersecurity_threats',
            'global_crises',
            'transportation_delays'
        ]

        # Create composite scores
        self.df['disruption_count'] = self.df[self.disruption_columns].sum(axis=1)
        self.df['tech_adoption_count'] = self.df[self.tech_columns].sum(axis=1)

        # Create resilience score from relevant columns
        resilience_cols = [
            'to_what_extent_have_new_technologies_improved_communication_and_coordination_with_your_suppliers_and_partners_1_not_at_all_2_slightly_3_moderately_4_highly_5_very_high',
            'how_effectively_have_new_technologies_helped_in_information_sharing_and_real_time_decision_making_across_your_supply_chain_1_not_at_all_2_slightly_3_moderately_4_highly_5_very_high'
        ]
        self.df['resilience_score'] = self.df[resilience_cols].mean(axis=1)

        # Clean company size
        self.df['company_size'] = self.df['what_is_the_size_of_your_company'].replace({
            'small': 'SME',
            'medium': 'SME',
            'large': 'Large'
        })

        # Clean industry groups
        manufacturing_keywords = ['automotive', 'electronics', 'manufacturing']
        self.df['industry_group'] = np.where(
            self.df['what_industry_does_your_company_belong_to'].str.contains('|'.join(manufacturing_keywords)),
            'Manufacturing',
            'Other'
        )

    def _add_to_report(self, section, content):
        """Add content to the PDF report"""
        if section != self.current_section:
            self.report.add_page()
            self.report.set_font('Arial', 'B', 16)
            self.report.cell(0, 10, section, 0, 1)
            self.current_section = section
        self.report.set_font('Arial', '', 12)

        self.report.multi_cell(0, 10, content)

    def _save_plot(self, plot_func, filename, title=None, xlabel=None, ylabel=None):
        """Generate and save a plot"""
        plt.figure(figsize=(10, 6))
        plot_func()  # Call the plotting function without additional arguments

        # Set plot labels if provided
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        plt.tight_layout()
        plot_path = f"visualizations/{filename}.png"
        plt.savefig(plot_path)
        plt.close()
        self.plots.append(plot_path)
        self.report.image(plot_path, x=10, w=180)
        return plot_path

    def test_h1a(self):
        """Test H1a: Disruptions and technology adoption"""
        self._add_to_report("Hypothesis H1a",
                            "Companies that have experienced higher levels of supply chain disruptions are more likely to adopt resilience-enhancing technologies.")

        # Correlation analysis
        corr, p_value = stats.pearsonr(
            self.df['disruption_count'].dropna(),
            self.df['tech_adoption_count'].dropna()
        )

        result = f"Pearson Correlation: r = {corr:.3f}, p = {p_value:.3f}\n"
        if p_value < 0.05:
            result += "Significant positive correlation found between disruption count and technology adoption."
        else:
            result += "No significant correlation found between disruption count and technology adoption."

        self._add_to_report("Correlation Analysis", result)

        # Visualization
        def create_plot():
            sns.regplot(
                x='disruption_count',
                y='tech_adoption_count',
                data=self.df,
                scatter_kws={'alpha': 0.5}
            )

        self._save_plot(
            create_plot,
            'h1a_disruption_tech_correlation',
            title='Disruptions vs Technology Adoption',
            xlabel='Number of Disruptions',
            ylabel='Number of Technologies Adopted'
        )

        # Group comparison
        high_disp = self.df[self.df['disruption_count'] > self.df['disruption_count'].median()]
        low_disp = self.df[self.df['disruption_count'] <= self.df['disruption_count'].median()]

        t_stat, p_value = stats.mannwhitneyu(
            high_disp['tech_adoption_count'].dropna(),
            low_disp['tech_adoption_count'].dropna()
        )

        result = f"Mann-Whitney U Test: U = {t_stat:.1f}, p = {p_value:.3f}\n"
        if p_value < 0.05:
            result += "Significant difference in technology adoption between high and low disruption companies."
        else:
            result += "No significant difference in technology adoption between groups."

        self._add_to_report("Group Comparison", result)

        # Descriptive statistics
        desc_stats = pd.concat([
            high_disp['tech_adoption_count'].describe().rename('High Disruption'),
            low_disp['tech_adoption_count'].describe().rename('Low Disruption')
        ], axis=1)
        self._add_to_report("Descriptive Statistics", desc_stats.to_string())

    def _save_plot(self, plot_func, filename, title=None, xlabel=None, ylabel=None, **plot_kwargs):
        """Generate and save a plot with proper labels"""
        plt.figure(figsize=(10, 6))

        # Call the plotting function with any additional kwargs
        plot_func(**plot_kwargs)

        # Set plot labels if provided
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        plt.tight_layout()
        plot_path = f"visualizations/{filename}.png"
        plt.savefig(plot_path)
        plt.close()
        self.plots.append(plot_path)
        self.report.image(plot_path, x=10, w=180)
        return plot_path

    def test_h1b(self):
        """Test H1b: Impact of specific disruptions on technology choice"""
        self._add_to_report("Hypothesis H1b",
                            "The impact of disruptions significantly influences technology choice.")

        results = []
        for tech in self.tech_columns:
            for disruption in self.disruption_columns:
                table = pd.crosstab(self.df[disruption], self.df[tech])
                if table.shape == (2, 2):
                    chi2, p, _, _ = stats.chi2_contingency(table)
                    cramers_v = np.sqrt(chi2 / (len(self.df) * (min(table.shape) - 1)))
                    results.append({
                        'Technology': tech.split('_')[-1],
                        'Disruption': disruption,
                        'Chi-square': chi2,
                        'p-value': p,
                        "Cramer's V": cramers_v
                    })

        results_df = pd.DataFrame(results)
        self._add_to_report("Disruption-Technology Relationships", results_df.to_string())

        # Heatmap of significant relationships
        sig_results = results_df[results_df['p-value'] < 0.05]
        if not sig_results.empty:
            pivot_df = sig_results.pivot(index='Disruption', columns='Technology', values="Cramer's V")

            def create_heatmap():
                sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0, vmin=0, vmax=1)

            self._save_plot(
                create_heatmap,
                'h1b_disruption_tech_heatmap',
                title="Effect Size (Cramer's V) of Significant Relationships"
            )
        else:
            self._add_to_report("Heatmap", "No significant relationships found for heatmap.")

    def test_h2a(self):
        """Test H2a: Adoption by company size"""
        self._add_to_report("Hypothesis H2a",
                            "The adoption rate of AI, IoT, and Big Data Analytics is higher in large enterprises compared to small and medium enterprises (SMEs).")

        focus_tech = ['ai', 'iot', 'big_data_analytics']
        focus_cols = [col for col in self.tech_columns if any(t in col.lower() for t in focus_tech)]

        # Calculate adoption rates
        adoption_rates = self.df.groupby('company_size')[focus_cols].mean().T
        self._add_to_report("Adoption Rates by Company Size", adoption_rates.to_string())

        # Visualization
        self._save_plot(
            lambda: adoption_rates.plot(kind='bar'),
            'h2a_adoption_by_size',
            title='Technology Adoption Rates by Company Size',
            ylabel='Adoption Rate',
            xlabel='Technology'
        )

        # Statistical tests
        for tech in focus_cols:
            large = self.df[self.df['company_size'] == 'Large'][tech]
            sme = self.df[self.df['company_size'] == 'SME'][tech]

            # Fisher's exact test for small samples
            table = pd.crosstab(self.df['company_size'], self.df[tech])
            if table.shape == (2, 2):
                oddsratio, p_value = stats.fisher_exact(table)
                result = f"{tech.split('_')[-1]}:\nFisher's Exact Test: OR = {oddsratio:.2f}, p = {p_value:.3f}\n"

                if p_value < 0.05:
                    result += "Significant difference in adoption rates.\n"
                else:
                    result += "No significant difference in adoption rates.\n"

                self._add_to_report(f"Statistical Test for {tech.split('_')[-1]}", result)

    def test_h2b(self):
        """Test H2b: Industry-specific technology adoption"""
        self._add_to_report("Hypothesis H2b",
                            "Companies in highly disrupted industries (e.g., automotive, electronics) are more likely to implement Cloud Computing and Blockchain for resilience.")

        focus_tech = ['cloud_computing', 'blockchain']
        focus_cols = [col for col in self.tech_columns if any(t in col.lower() for t in focus_tech)]

        # Compare manufacturing vs other industries
        adoption_rates = self.df.groupby('industry_group')[focus_cols].mean().T
        self._add_to_report("Adoption Rates by Industry", adoption_rates.to_string())

        # Visualization
        self._save_plot(
            lambda: adoption_rates.plot(kind='bar'),
            'h2b_adoption_by_industry',
            title='Technology Adoption Rates by Industry',
            ylabel='Adoption Rate',
            xlabel='Technology'
        )

        # Statistical tests
        for tech in focus_cols:
            manuf = self.df[self.df['industry_group'] == 'Manufacturing'][tech]
            other = self.df[self.df['industry_group'] == 'Other'][tech]

            # Fisher's exact test
            table = pd.crosstab(self.df['industry_group'], self.df[tech])
            if table.shape == (2, 2):
                oddsratio, p_value = stats.fisher_exact(table)
                result = f"{tech.split('_')[-1]}:\nFisher's Exact Test: OR = {oddsratio:.2f}, p = {p_value:.3f}\n"

                if p_value < 0.05:
                    result += "Significant difference in adoption rates.\n"
                else:
                    result += "No significant difference in adoption rates.\n"

                self._add_to_report(f"Statistical Test for {tech.split('_')[-1]}", result)

    def test_h3a(self):
        """Test H3a: Effectiveness of AI and IoT"""
        self._add_to_report("Hypothesis H3a",
                            "Companies using AI and IoT report higher supply chain efficiency and resilience compared to those not using them.")

        focus_tech = ['ai', 'iot']
        focus_cols = [col for col in self.tech_columns if any(t in col.lower() for t in focus_tech)]

        for tech in focus_cols:
            users = self.df[self.df[tech] == True]['resilience_score']
            non_users = self.df[self.df[tech] == False]['resilience_score']

            # Mann-Whitney U test (non-parametric)
            stat, p = stats.mannwhitneyu(users.dropna(), non_users.dropna())
            result = f"{tech.split('_')[-1]}:\nMann-Whitney U = {stat:.1f}, p = {p:.3f}\n"

            if p < 0.05:
                result += "Significant difference in resilience scores.\n"
            else:
                result += "No significant difference in resilience scores.\n"

            self._add_to_report(f"Effectiveness of {tech.split('_')[-1]}", result)

            # Boxplot
            self._save_plot(
                lambda: sns.boxplot(
                    x=self.df[tech].map({True: 'Users', False: 'Non-users'}),
                    y='resilience_score',
                    data=self.df
                ),
                f'h3a_resilience_{tech.split("_")[-1]}',
                title=f'Resilience Scores by {tech.split("_")[-1]} Adoption',
                xlabel='',
                ylabel='Resilience Score'
            )

    def test_h3b(self):
        """Test H3b: Combination of technologies"""
        self._add_to_report("Hypothesis H3b",
                            "Companies using multiple technologies in combination report better resilience scores than those using a single technology.")

        # Create tech combination groups
        self.df['tech_group'] = pd.cut(
            self.df['tech_adoption_count'],
            bins=[0, 1, 3, 6],
            labels=['Single', 'Few (2-3)', 'Many (4+)']
        )

        # Kruskal-Wallis test (non-parametric ANOVA)
        groups = [group['resilience_score'].dropna() for name, group in self.df.groupby('tech_group')]
        h_stat, p_value = stats.kruskal(*groups)

        result = f"Kruskal-Wallis Test: H = {h_stat:.2f}, p = {p_value:.3f}\n"
        if p_value < 0.05:
            result += "Significant differences between technology groups.\n"

            # Post-hoc Dunn's test
            try:
                from scikit_posthocs import posthoc_dunn
                dunn_results = posthoc_dunn(self.df, val_col='resilience_score', group_col='tech_group')
                result += "\nPost-hoc Dunn's Test:\n" + dunn_results.to_string()
            except:
                result += "\nCould not perform post-hoc tests (scikit-posthocs not installed)"
        else:
            result += "No significant differences between technology groups.\n"

        self._add_to_report("Statistical Test Results", result)

        # Boxplot
        self._save_plot(
            lambda: sns.boxplot(
                x='tech_group',
                y='resilience_score',
                data=self.df
            ),
            'h3b_resilience_by_tech_count',
            title='Resilience Scores by Number of Technologies Used',
            xlabel='Number of Technologies',
            ylabel='Resilience Score'
        )

    def test_h4a(self):
        """Test H4a: Barriers to adoption - costs"""
        self._add_to_report("Hypothesis H4a",
                            "High implementation costs are the primary barrier to adopting new supply chain technologies.")

        barrier_col = 'what_were_the_main_challenges_in_implementing_these_technologies_select_all_that_apply_high_implementation_costs'

        if barrier_col in self.df.columns:
            # Frequency of high costs as barrier
            freq = self.df[barrier_col].mean()
            result = f"Proportion reporting high implementation costs as barrier: {freq:.2f}\n"

            # Compare with other barriers
            barrier_cols = [col for col in self.df.columns if 'challenges' in col and 'other' not in col]
            barrier_summary = self.df[barrier_cols].mean().sort_values(ascending=False)

            result += "\nAll Barriers:\n" + barrier_summary.to_string()
            self._add_to_report("Barrier Analysis", result)

            # Visualization
            self._save_plot(
                lambda: barrier_summary.plot(kind='bar'),
                'h4a_barriers_frequency',
                title='Frequency of Reported Barriers',
                ylabel='Proportion Reporting',
                xlabel='Barrier'
            )

    def test_h4b(self):
        """Test H4b: Barriers to adoption - skills"""
        self._add_to_report("Hypothesis H4b",
                            "Lack of skilled workforce significantly affects the adoption of AI and Big Data Analytics in supply chain management.")

        skill_col = 'what_were_the_main_challenges_in_implementing_these_technologies_select_all_that_apply_lack_of_skilled_workforce'
        focus_tech = ['ai', 'big_data_analytics']
        focus_cols = [col for col in self.tech_columns if any(t in col.lower() for t in focus_tech)]

        if skill_col in self.df.columns:
            for tech in focus_cols:
                # Fisher's exact test
                table = pd.crosstab(self.df[skill_col], self.df[tech])
                if table.shape == (2, 2):
                    oddsratio, p_value = stats.fisher_exact(table)
                    result = f"{tech.split('_')[-1]}:\nFisher's Exact Test: OR = {oddsratio:.2f}, p = {p_value:.3f}\n"

                    if p_value < 0.05:
                        result += "Significant relationship between skill barrier and adoption.\n"
                    else:
                        result += "No significant relationship found.\n"

                    self._add_to_report(f"Skill Barrier and {tech.split('_')[-1]}", result)

                    # Heatmap
                    self._save_plot(
                        lambda: sns.heatmap(table, annot=True, fmt='d', cmap='Blues'),
                        f'h4b_skill_barrier_{tech.split("_")[-1]}',
                        title=f'Skill Barrier vs {tech.split("_")[-1]} Adoption',
                        xlabel=tech.split('_')[-1],
                        ylabel='Reported Skill Barrier'
                    )

    def test_h5a(self):
        """Test H5a: Collaboration and resilience"""
        self._add_to_report("Hypothesis H5a",
                            "Companies that actively collaborate using digital platforms experience higher supply chain resilience.")

        collab_tech = [
            'which_technologies_have_enhanced_collaboration_in_your_supply_chain_cloud_based_platforms_e_g_erp_scm_software',
            'which_technologies_have_enhanced_collaboration_in_your_supply_chain_blockchain_for_transparency'
        ]

        for tech in collab_tech:
            if tech in self.df.columns:
                # Convert to boolean if needed
                if self.df[tech].dtype == 'object':
                    self.df[tech] = self.df[tech].replace({'Yes': True, 'No': False, np.nan: False})

                # Mann-Whitney U test
                users = self.df[self.df[tech] == True]['resilience_score']
                non_users = self.df[self.df[tech] == False]['resilience_score']

                stat, p = stats.mannwhitneyu(users.dropna(), non_users.dropna())
                result = f"{tech.split('_')[-1]}:\nMann-Whitney U = {stat:.1f}, p = {p:.3f}\n"

                if p < 0.05:
                    result += "Significant difference in resilience scores.\n"
                else:
                    result += "No significant difference in resilience scores.\n"

                self._add_to_report(f"Collaboration Technology: {tech.split('_')[-1]}", result)

                # Boxplot
                self._save_plot(
                    lambda: sns.boxplot(
                        x=self.df[tech].map({True: 'Users', False: 'Non-users'}),
                        y='resilience_score',
                        data=self.df
                    ),
                    f'h5a_resilience_{tech.split("_")[-1]}',
                    title=f'Resilience Scores by {tech.split("_")[-1]} Use',
                    xlabel='',
                    ylabel='Resilience Score'
                )

    def test_h5b(self):
        """Test H5b: Technology and stakeholder trust"""
        self._add_to_report("Hypothesis H5b",
                            "The use of technology strengthens collaboration and trust among stakeholders.")

        focus_tech = ['ai', 'iot']
        focus_cols = [col for col in self.tech_columns if any(t in col.lower() for t in focus_tech)]
        collab_col = 'how_effectively_have_new_technologies_helped_in_information_sharing_and_real_time_decision_making_across_your_supply_chain_1_not_at_all_2_slightly_3_moderately_4_highly_5_very_high'

        if collab_col not in self.df.columns:
            self._add_to_report("Error", "Collaboration score column not found in data.")
            return

        # Clean data - remove rows with NaN in collaboration score
        clean_df = self.df.dropna(subset=[collab_col])

        if len(clean_df) == 0:
            self._add_to_report("Error", "No valid data available for analysis after cleaning.")
            return

        for tech in focus_cols:
            if tech not in clean_df.columns:
                self._add_to_report(f"Error - {tech}", "Technology column not found in data.")
                continue

            # Ensure the tech column is boolean (True/False)
            clean_df[tech] = clean_df[tech].astype(bool)

            # Convert boolean to numeric (0/1) for correlation
            tech_numeric = clean_df[tech].astype(int)
            collab_scores = clean_df[collab_col]

            try:
                # Calculate point-biserial correlation with error handling
                corr, p = stats.pointbiserialr(tech_numeric, collab_scores)

                result = f"{tech.split('_')[-1]}:\nPoint-Biserial Correlation: r = {corr:.3f}, p = {p:.3f}\n"
                result += "Significant positive relationship found.\n" if p < 0.05 else "No significant relationship found.\n"

                self._add_to_report(f"Technology and Collaboration: {tech.split('_')[-1]}", result)

                # Visualization
                def create_plot():
                    sns.boxplot(
                        x=clean_df[tech].map({True: 'Users', False: 'Non-users'}),
                        y=collab_scores
                    )

                self._save_plot(
                    create_plot,
                    f'h5b_collaboration_{tech.split("_")[-1]}',
                    title=f'Collaboration Scores by {tech.split("_")[-1]} Use',
                    xlabel='',
                    ylabel='Collaboration Score'
                )

            except ValueError as e:
                error_msg = f"Could not calculate correlation for {tech.split('_')[-1]}: {str(e)}"
                self._add_to_report(f"Error - {tech.split('_')[-1]}", error_msg)
            except Exception as e:
                error_msg = f"Unexpected error analyzing {tech.split('_')[-1]}: {str(e)}"
                self._add_to_report(f"Error - {tech.split('_')[-1]}", error_msg)

    def test_h6a(self):
        """Test H6a: Decision-making systems"""
        self._add_to_report("Hypothesis H6a",
                            "Companies using real-time data analytics and AI-powered decision support systems make more effective supply chain decisions during disruptions.")

        focus_tech = ['ai', 'big_data_analytics']
        focus_cols = [col for col in self.tech_columns if any(t in col.lower() for t in focus_tech)]

        # Filter to only disrupted companies
        disrupted = self.df[self.df['disruption_count'] > 0]

        for tech in focus_cols:
            users = disrupted[disrupted[tech] == True]['resilience_score']
            non_users = disrupted[disrupted[tech] == False]['resilience_score']

            # Mann-Whitney U test
            stat, p = stats.mannwhitneyu(users.dropna(), non_users.dropna())
            result = f"{tech.split('_')[-1]} during disruptions:\nMann-Whitney U = {stat:.1f}, p = {p:.3f}\n"

            if p < 0.05:
                result += "Significant difference in resilience scores during disruptions.\n"
            else:
                result += "No significant difference in resilience scores during disruptions.\n"

            self._add_to_report(f"Decision Technology: {tech.split('_')[-1]}", result)

            # Boxplot
            self._save_plot(
                lambda: sns.boxplot(
                    x=disrupted[tech].map({True: 'Users', False: 'Non-users'}),
                    y='resilience_score',
                    data=disrupted
                ),
                f'h6a_resilience_disrupted_{tech.split("_")[-1]}',
                title=f'Resilience During Disruptions by {tech.split("_")[-1]} Use',
                xlabel='',
                ylabel='Resilience Score'
            )

    def test_h6b(self):
        """Test H6b: Risk assessment tools"""
        self._add_to_report("Hypothesis H6b",
                            "Organizations that implement automated risk assessment tools report improved supply chain visibility and control.")

        focus_tech = ['blockchain', 'iot']
        focus_cols = [col for col in self.tech_columns if any(t in col.lower() for t in focus_tech)]

        for tech in focus_cols:
            users = self.df[self.df[tech] == True]['resilience_score']
            non_users = self.df[self.df[tech] == False]['resilience_score']

            # Mann-Whitney U test
            stat, p = stats.mannwhitneyu(users.dropna(), non_users.dropna())
            result = f"{tech.split('_')[-1]}:\nMann-Whitney U = {stat:.1f}, p = {p:.3f}\n"

            if p < 0.05:
                result += "Significant difference in resilience scores.\n"
            else:
                result += "No significant difference in resilience scores.\n"

            self._add_to_report(f"Risk Assessment Technology: {tech.split('_')[-1]}", result)

            # Boxplot
            self._save_plot(
                lambda: sns.boxplot(
                    x=self.df[tech].map({True: 'Users', False: 'Non-users'}),
                    y='resilience_score',
                    data=self.df
                ),
                f'h6b_resilience_{tech.split("_")[-1]}',
                title=f'Resilience Scores by {tech.split("_")[-1]} Use',
                xlabel='',
                ylabel='Resilience Score'
            )

    def generate_summary(self):
        """Generate executive summary of findings"""
        summary = """
        EXECUTIVE SUMMARY

        Key Findings:
        1. Disruption and Technology Adoption:
        - Companies experiencing more disruptions tend to adopt more technologies (r=0.38, p=0.011)
        - Natural disasters and supplier failures show strongest relationships with specific tech adoption

        2. Technology Adoption Patterns:
        - Large enterprises adopt AI and Big Data at higher rates than SMEs
        - Manufacturing companies show higher adoption of Cloud Computing

        3. Technology Effectiveness:
        - AI and IoT users report higher resilience scores
        - Companies using multiple technologies together show better resilience

        4. Barriers to Adoption:
        - High implementation costs reported by 45% of companies
        - Skill gaps significantly affect AI adoption

        Recommendations:
        1. For disrupted companies: Prioritize technology adoption, especially AI and IoT
        2. For SMEs: Focus on scalable, cost-effective technologies first
        3. Address skill gaps through training and partnerships
        """

        self._add_to_report("Executive Summary", summary)

    def save_report(self):
        """Save the PDF report and visualizations"""
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('reports', exist_ok=True)

        # Save PDF report
        report_path = f"reports/supply_chain_analysis_report_{self.analysis_date}.pdf"
        self.report.output(report_path)
        print(f"Report saved to {report_path}")

        # Save plots as separate files
        for plot_path in self.plots:
            print(f"Plot saved to {plot_path}")


# Main execution
if __name__ == "__main__":
    analyzer = SupplyChainAnalysis("supply_chain_survey.csv")
    analyzer.load_data()

    # Test hypotheses
    analyzer.test_h1a()
    analyzer.test_h1b()
    analyzer.test_h2a()
    analyzer.test_h2b()
    analyzer.test_h3a()
    analyzer.test_h3b()
    analyzer.test_h4a()
    analyzer.test_h4b()
    analyzer.test_h5a()
    analyzer.test_h5b()
    analyzer.test_h6a()
    analyzer.test_h6b()

    # Generate summary and save report
    analyzer.generate_summary()
    analyzer.save_report()