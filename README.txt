# Dataset Title: Supply Chain Technologies
# Author: RENJIN RAJU
# DOI: [Paste your Dataverse DOI here once published]
# Date of Publication: [Enter the final publication date]
# Institution: Deggendorf Institute of Technology
# Thesis Project: Master’s Thesis – Technologies Enhancing Supply Chain Resilience

## Overview:
This dataset contains all the supporting materials for the thesis project investigating the role of digital technologies in enhancing Supply Chain Resilience (SCR). It includes raw and cleaned survey data, Python scripts for statistical analysis and visualization, and output files used in the final report.

---

## File Descriptions:

### 📊 Survey Data:
- **results-survey958981.csv**
  - Exported raw data from LimeSurvey.
  - Includes partial and complete responses with internal variable names.
  - Not suitable for direct analysis.

- **survey data cleaned.csv**
  - Preprocessed dataset used for statistical and visual analysis.
  - Missing values handled, columns renamed, and irrelevant responses filtered.

---

### 🧠 Python Scripts:
- **cronbach's alpha.py**
  - Computes internal consistency using Cronbach’s Alpha.
  - Used for construct reliability validation.

- **data distribution.py**
  - Tests data normality using Shapiro-Wilk and visual distribution checks.

- **Survey Analysis Final.py**
  - Master script used for all core statistical analysis:
    - Mann-Whitney U
    - Kruskal-Wallis
    - Chi-square tests
    - Pearson correlation
    - Bar/box plots and heatmaps

- **updated scenario modelling.py**
  - Applies Bayesian scenario modelling and co-adoption rule mining (association rule mining using `mlxtend`).

---

### 📄 Output Files:
- **Supply_Chain_Resilience_Analysis_Report.pdf**
  - Consolidated plots and visuals generated from the scripts.
  - Includes interpretation-ready visualizations for inclusion in thesis.

---

## Data Usage License:
All files are released under **CC0 1.0 (Public Domain Dedication)**.
This means the dataset is free to use for academic and research purposes with proper citation.

---

## How to Use:
1. Download the cleaned data and analysis scripts.
2. Use Python 3.12+ with libraries listed in the scripts (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `sklearn`, `mlxtend`).
3. Open and run the scripts in sequence for reproducibility.
4. Visuals generated are saved in PDF or PNG formats as per the script configuration.

---

## Citation:
If you use this dataset, please cite as:

RAJU, RENJIN. 2025. “Supply Chain Technologies.” Harvard Dataverse. https://doi.org/10.7910/DVN/XXXXXX
