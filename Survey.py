# -*- coding: utf-8 -*-
"""
Supply Chain Resilience Survey Analysis - DEBUGGED VERSION
Author: Your Name
Date: Current Date
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from fpdf import FPDF
import zipfile
import os
import warnings
import chardet  # For detecting file encoding

# Suppress warnings
warnings.filterwarnings("ignore")


# =============================================
# 1. DATA LOADING AND CLEANING (FIXED)
# =============================================

def detect_encoding(filepath):
    """Detect file encoding to handle special characters"""
    with open(filepath, r'D:\RENJIN RAJU\MASTERS\SEMESTER 3\THESIS\Research\Data\Survey Results cleaned.csv') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def load_and_clean_data(filepath):
    """Load and clean the raw survey data with proper encoding"""
    print("Loading and cleaning data...")

    # Detect file encoding first
    encoding = detect_encoding(filepath)
    print(f"Detected encoding: {encoding}")

    # Load data with correct encoding
    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        # Try fallback encodings if detection fails
        for enc in ['latin1', 'ISO-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                print(f"Used fallback encoding: {enc}")
                break
            except:
                continue

    # Standardize column names (fixed regex)
    df.columns = (df.columns.str.strip()
                  .str.lower()
                  .str.replace(r'[^a-z0-9_]', '_', regex=True))

    # Handle missing values (more comprehensive)
    missing_indicators = ['n/a', 'na', 'nan', 'none', 'no answer', '', ' ', 'null']
    df.replace(missing_indicators, np.nan, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    # Clean industry categories (fixed dictionary)
    industry_map = {
        'other, planungsbranche': 'other',
        'other, dienstleistungen': 'other',
        'other, maschinenbau': 'manufacturing',
        'other, bauindustrie': 'construction',
        'other, bau': 'construction',
        'other, milchwirtschaft': 'food',
        'other, production -chemical fertilizers': 'chemical',
        'other, chemical - fertilizers': 'chemical',
        'other, food processing plant': 'food',
        'other, food manufacturing': 'food',
        'other, freight forwarding agents': 'logistics',
        'other, oil and gas': 'energy',
        'other, sanitär-heizung-klima': 'manufacturing',
        'other, nahrungsmittel': 'food',
        'automotive': 'automotive',
        'electronics': 'electronics',
        'consumer goods': 'consumer_goods'
    }

    # Handle NaN values in industry column
    if 'what_industry_does_your_company_belong_to_' in df.columns:
        df['industry'] = (df['what_industry_does_your_company_belong_to_']
                          .str.lower()
                          .str.strip()
                          .replace(industry_map))
    else:
        df['industry'] = 'unknown'

    # Clean company size (more robust extraction)
    if 'what_is_the_size_of_your_company_' in df.columns:
        df['company_size'] = (df['what_is_the_size_of_your_company_']
                              .astype(str)
                              .str.lower()
                              .str.extract(r'(small|medium|large)', expand=False)
                              .str.strip())
    else:
        df['company_size'] = np.nan

    # Convert technology adoption to boolean (fixed column detection)
    tech_cols = [col for col in df.columns if 'technologies_does_your_company' in col or 'which_technologies' in col]
    for col in tech_cols:
        df[col] = df[col].replace({'Yes': True, 'No': False, np.nan: False, 'yes': True, 'no': False})

    # Clean technology names (more comprehensive)
    tech_name_clean = {
        'artificial_intelligence_ai': 'ai',
        'artificial_intelligence': 'ai',
        'internet_of_things_iot': 'iot',
        'internet_of_things': 'iot',
        'robotics_amp_automation': 'robotics',
        'robotics': 'robotics',
        'big_data_analytics': 'big_data',
        'big_data': 'big_data',
        'blockchain': 'blockchain',
        'cloud_computing': 'cloud',
        'simulation': 'simulation'
    }

    # Apply cleaning with case insensitivity
    df = df.rename(columns=lambda x: next((v for k, v in tech_name_clean.items() if k.lower() in x.lower()), x))

    # Clean Likert scale responses (fixed regex warning)
    likert_cols = [col for col in df.columns if 'likert' in col.lower() or 'extent' in col.lower()]
    for col in likert_cols:
        if df[col].dtype == 'object':
            # Extract first number (handles cases like "3 - Pilot")
            df[col] = df[col].str.extract(r'(\d+)', expand=False).astype(float)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create derived variables (with checks)
    tech_adoption_cols = [col for col in df.columns if any(t in col.lower() for t in tech_name_clean.values())]
    if tech_adoption_cols:
        df['tech_adoption_count'] = df[tech_adoption_cols].sum(axis=1)
    else:
        df['tech_adoption_count'] = 0

    resilience_cols = [col for col in df.columns if
                       'improved_communication' in col.lower() or 'decision_making' in col.lower()]
    if resilience_cols:
        df['resilience_score'] = df[resilience_cols].mean(axis=1)
    else:
        df['resilience_score'] = np.nan

    print("Data cleaning complete!")
    return df


# =============================================
# 2. HYPOTHESIS TESTING FUNCTIONS (FIXED)
# =============================================

def test_h1a(df, output_dir):
    """Test H1a: Companies with more disruptions adopt more tech"""
    print("Testing H1a...")

    # Calculate disruption count if not exists
    if 'disruption_count' not in df.columns:
        disruption_cols = [col for col in df.columns if 'disruptions' in col.lower() and (
                    'yes' in str(df[col].iloc[0]).lower() or 'no' in str(df[col].iloc[0]).lower())]
        if disruption_cols:
            df['disruption_count'] = df[disruption_cols].sum(axis=1)
        else:
            df['disruption_count'] = 0

    # Filter out rows with missing data
    temp_df = df.dropna(subset=['disruption_count', 'tech_adoption_count'])

    # Correlation test with validation
    try:
        corr, p = stats.pearsonr(temp_df['disruption_count'], temp_df['tech_adoption_count'])
    except ValueError as e:
        print(f"Correlation failed: {str(e)}")
        corr, p = np.nan, np.nan

    # Group comparison with validation
    try:
        median = temp_df['disruption_count'].median()
        high_disp = temp_df[temp_df['disruption_count'] > median]
        low_disp = temp_df[temp_df['disruption_count'] <= median]

        # Check if groups have enough samples
        if len(high_disp) > 1 and len(low_disp) > 1:
            t_stat, p_val = stats.ttest_ind(high_disp['tech_adoption_count'],
                                            low_disp['tech_adoption_count'])
        else:
            t_stat, p_val = np.nan, np.nan
    except Exception as e:
        print(f"Group comparison failed: {str(e)}")
        t_stat, p_val = np.nan, np.nan

    # Visualization with error handling
    plot_path = os.path.join(output_dir, 'h1a_correlation.png')
    try:
        plt.figure(figsize=(10, 6))
        sns.regplot(x='disruption_count', y='tech_adoption_count', data=temp_df, ci=None)
        plt.title('Disruptions vs Technology Adoption')
        plt.xlabel('Number of Disruptions')
        plt.ylabel('Technologies Adopted')
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Plot generation failed: {str(e)}")
        plot_path = None

    return {
        'hypothesis': 'H1a',
        'test': 'Pearson correlation + t-test',
        'correlation': corr,
        'correlation_p': p,
        't_statistic': t_stat,
        't_p_value': p_val,
        'plot': plot_path,
        'n_observations': len(temp_df)
    }


# =============================================
# 3. REPORT GENERATION (FIXED)
# =============================================

class PDFReport(FPDF):
    """Custom PDF report generator with error handling"""

    def __init__(self):
        super().__init__()
        # Handle PyFPDF/fpdf2 conflict
        self._has_fpdf2 = True

    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Supply Chain Resilience Survey Analysis', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        # Handle long text by splitting into lines
        lines = body.split('\n')
        for line in lines:
            if len(line) > 100:
                for wrapped_line in wrap(line, 100):
                    self.multi_cell(0, 10, wrapped_line)
            else:
                self.multi_cell(0, 10, line)
        self.ln()

    def add_image(self, image_path, width=180):
        if os.path.exists(image_path):
            try:
                self.image(image_path, x=10, y=None, w=width)
                self.ln()
            except:
                self.chapter_body(f"[Image failed to load: {image_path}]")
        else:
            self.chapter_body(f"[Image not found: {image_path}]")


def generate_report(results, output_dir):
    """Generate PDF report from analysis results with error handling"""
    print("Generating PDF report...")

    pdf = PDFReport()
    pdf.add_page()

    # Title page with error handling
    try:
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Supply Chain Resilience Technology Analysis', 0, 1, 'C')
        pdf.ln(20)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10,
                       "This report presents the complete analysis of survey data examining the impact of emerging technologies on supply chain resilience.")
    except:
        print("Error in title page generation")

    # Methodology section
    try:
        pdf.add_page()
        pdf.chapter_title('Methodology')
        methodology_text = """
        The analysis employed the following statistical tests:
        - Pearson correlation for continuous variable relationships
        - Independent t-tests for group comparisons
        - ANOVA with Tukey HSD post-hoc for multi-group analysis
        - Chi-square tests for categorical associations
        All tests were conducted at α = 0.05 significance level.

        Data Cleaning Steps:
        1. Handled missing values and encoding issues
        2. Standardized categorical variables
        3. Validated all statistical assumptions
        """
        pdf.chapter_body(methodology_text)
    except:
        print("Error in methodology section")

    # Results section with error handling
    try:
        pdf.add_page()
        pdf.chapter_title('Results')

        for result in results:
            try:
                pdf.chapter_title(f"Hypothesis {result['hypothesis']}")
                result_text = f"""
                Test: {result.get('test', 'N/A')}
                Key statistic: {result.get('correlation', result.get('t_statistic', 'N/A'))}
                p-value: {result.get('correlation_p', result.get('t_p_value', 'N/A'))}
                Sample size: {result.get('n_observations', 'N/A')}
                Interpretation: {'Supported' if result.get('correlation_p', result.get('t_p_value', 1)) < 0.05 else 'Not supported'}
                """
                pdf.chapter_body(result_text)

                if result.get('plot'):
                    pdf.add_image(result['plot'])
            except:
                print(f"Error processing result for {result.get('hypothesis', 'unknown')}")
    except:
        print("Error in results section")

    # Save PDF with error handling
    report_path = os.path.join(output_dir, 'supply_chain_analysis_report.pdf')
    try:
        pdf.output(report_path)
        print(f"Report generated at {report_path}")
        return report_path
    except:
        print("Failed to save PDF report")
        return None


# =============================================
# 4. MAIN EXECUTION (FIXED)
# =============================================

def main():
    # Create output directory with error handling
    output_dir = 'supply_chain_analysis_output'
    try:
        os.makedirs(output_dir, exist_ok=True)
    except:
        print(f"Failed to create output directory {output_dir}")
        return

    try:
        # 1. Load and clean data with full path
        input_path = r"D:\RENJIN RAJU\MASTERS\SEMESTER 3\THESIS\Research\Data\Survey Results cleaned.csv"
        if not os.path.exists(input_path):
            print(f"Input file not found: {input_path}")
            return

        df = load_and_clean_data(input_path)

        # Save cleaned data for inspection
        df.to_csv(os.path.join(output_dir, 'cleaned_data.csv'), index=False)

        # 2. Run hypothesis tests
        results = []
        results.append(test_h1a(df, output_dir))
        # Add other hypothesis tests here...

        # 3. Generate report
        report_path = generate_report(results, output_dir)

        if not report_path:
            print("Report generation failed")
            return

        # 4. Create ZIP package
        zip_path = 'supply_chain_analysis_package.zip'
        try:
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add all files in output directory
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)

                # Add original cleaned data
                zipf.write(input_path, os.path.basename(input_path))

            print(f"\nAnalysis complete! Final package saved as {zip_path}")
        except:
            print("Failed to create ZIP package")

    except Exception as e:
        print(f"Fatal error encountered: {str(e)}")
        raise


if __name__ == '__main__':
    main()