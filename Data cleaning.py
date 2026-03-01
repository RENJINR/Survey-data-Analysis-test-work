# Prepare Python script for analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
from wordcloud import WordCloud
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv(r"D:\\RENJIN RAJU\\MASTERS\\SEMESTER 3\\THESIS\\Research\\Analysis\\Survey data cleaned.csv", encoding='ISO-8859-1')

# ======= 1. Basic Preprocessing =======

# Encode categorical variables
cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])

# Drop columns with excessive missing values
df = df.drop(columns=[col for col in df.columns if df[col].isnull().sum() > 10])

# Fill remaining NaNs with mode (for categorical) or median (for numeric)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# ======= 2. Exploratory Visualizations =======

# Disruption sources
disruption_cols = [col for col in df.columns if col.startswith('Disruption_')]
df[disruption_cols].sum().plot(kind='bar', title='Disruption Sources')
plt.tight_layout()
plt.show()

# Responses taken
response_cols = [col for col in df.columns if col.startswith('response_') and 'count' not in col]
df[response_cols].sum().plot(kind='bar', title='Response Strategies')
plt.tight_layout()
plt.show()

# Technology adoption
adopt_cols = [col for col in df.columns if col.startswith('adoption_') and 'count' not in col]
df[adopt_cols].sum().plot(kind='bar', title='Technology Adoption')
plt.tight_layout()
plt.show()

# Barriers to adoption
barrier_cols = [col for col in df.columns if col.startswith('barriers_')]
df[barrier_cols].sum().plot(kind='bar', title='Adoption Barriers')
plt.tight_layout()
plt.show()

# ======= 3. Regression Analysis (Example) =======

# Predicting if AI was implemented based on barriers and disruptions
X = df[barrier_cols + disruption_cols]
y = df['implemented_AI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("=== Logistic Regression: Implemented AI ===")
print(classification_report(y_test, y_pred))

# ======= 4. Correlation Matrix =======

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# ======= 5. Cross-links: Adoption vs Barriers =======

adoption_sum = df[adopt_cols].sum(axis=1)
df['adoption_sum'] = adoption_sum
plt.scatter(df['adoption_sum'], df[barrier_cols].sum(axis=1))
plt.xlabel("Number of Technologies Adopted")
plt.ylabel("Number of Barriers Perceived")
plt.title("Technology Adoption vs Perceived Barriers")
plt.tight_layout()
plt.show()

# ======= Multivariate Logistic Regression =======

barrier_cols = [col for col in df.columns if 'barriers_' in col]
disruption_cols = [col for col in df.columns if 'Disruption_' in col]
adoption_cols = [col for col in df.columns if 'adoption_' in col and 'count' not in col]

features = barrier_cols + disruption_cols + adoption_cols
X = df[features]
y = df['implemented_Cloud_Computing']  # Change target here for other models

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== Multivariate Logistic Regression: Implemented Cloud Computing ===")
print(classification_report(y_test, y_pred))

# ======= K-Means Clustering for Technology Adoption Patterns =======

adoption_matrix = df[adoption_cols]
scaler = StandardScaler()
adoption_scaled = scaler.fit_transform(adoption_matrix)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(adoption_scaled)

# PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(adoption_scaled)
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', title='Clusters of Technology Adoption')
fig.show()

# ======= Plotly Dash-Like Output for Overview =======

fig1 = px.bar(df[disruption_cols].sum().reset_index(), x='index', y=0, title='Disruption Sources')
fig1.show()

fig2 = px.bar(df[barrier_cols].sum().reset_index(), x='index', y=0, title='Barriers to Adoption')
fig2.show()

fig3 = px.bar(df[adoption_cols].sum().reset_index(), x='index', y=0, title='Technology Adoption')
fig3.show()


insight_texts = [
    "Make machines as work in more than one power.",
    "The fast decision making using new technology and AI can solve most of the resiliences.",
    "Advanced Forecasting and Reshoring Tools, Sustainability Technology",
    "3d printing",
    "can eradicate or limit the damage and error though and limit the depreciation in product value to enhance the Gross domestic product"
]

insight_text_combined = " ".join(insight_texts)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(insight_text_combined)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Further Insights')
plt.tight_layout()
plt.show()


# ## ----- Summarized Rankings: Effective Technologies -----
# rank_cols_tech = [
#     'effective_technology_Rank_1', 'effective_technology_Rank_2', 'effective_technology_Rank_3',
#     'effective_technology_Rank_4', 'effective_technology_Rank_5', 'effective_technology_Rank_6',
#     'effective_technology_Rank_7'
# ]
#
# df[rank_cols_tech] = df[rank_cols_tech].astype(str)
# tech_summary = pd.DataFrame()
#
# for i, col in enumerate(rank_cols_tech, 1):
#     tech_summary = pd.concat([
#         tech_summary,
#         df[col].value_counts().rename(f"Rank_{i}")
#     ], axis=1)
#
# tech_summary = tech_summary.fillna(0).astype(int)
# tech_summary["Total"] = tech_summary.sum(axis=1)
# tech_summary = tech_summary.drop(index=["nan", "NaN", "None"], errors="ignore")  # Remove invalid entries
# tech_summary = tech_summary.sort_values("Total", ascending=True)
#
# # ----- Summarized Rankings: Visibility & Control Objectives -----
# rank_cols_control = [
#     'visibility_and_control_Rank_1', 'visibility_and_control_Rank_2', 'visibility_and_control_Rank_3',
#     'visibility_and_control_Rank_4', 'visibility_and_control_Rank_5', 'visibility_and_control_Rank_6'
# ]
#
# df[rank_cols_control] = df[rank_cols_control].astype(str)
# control_summary = pd.DataFrame()
#
# for i, col in enumerate(rank_cols_control, 1):
#     control_summary = pd.concat([
#         control_summary,
#         df[col].value_counts().rename(f"Rank_{i}")
#     ], axis=1)
#
# control_summary = control_summary.fillna(0).astype(int)
# control_summary["Total"] = control_summary.sum(axis=1)
# control_summary = control_summary.drop(index=["nan", "NaN", "None"], errors="ignore")
# control_summary = control_summary.sort_values("Total", ascending=True)
#
# plt.figure(figsize=(10, 6))
# sns.barplot(
#     x=tech_summary["Total"],
#     y=tech_summary.index.str.replace('_', ' ', regex=False).str.title(),
#     palette="crest"
# )
# plt.title("Total Mentions of Effective Technologies Across All Ranks", fontsize=14)
# plt.xlabel("Total Mentions", fontsize=12)
# plt.ylabel("Technology", fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(12, 7))
# sns.barplot(
#     x=control_summary["Total"],
#     y=control_summary.index.str.replace('_', ' ', regex=False).str.title(),
#     palette="flare"
# )
# plt.title("Total Mentions of Visibility and Control Objectives Across All Ranks", fontsize=14)
# plt.xlabel("Total Mentions", fontsize=12)
# plt.ylabel("Control Objective", fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
# plt.show()
#
#
# print("=== Effective Technology Rank Summary ===")
# print(tech_summary)
#
# print("\n=== Visibility & Control Objective Rank Summary ===")
# print(control_summary)
#
# tech_summary.to_csv("effective_technology_ranking_summary.csv")
# control_summary.to_csv("visibility_control_summary.csv")
# print("\n✅ Summaries saved as CSV files.")

# ========== EFFECTIVE TECHNOLOGIES SUMMARY ==========
rank_cols_tech = [
    'effective_technology_Rank_1', 'effective_technology_Rank_2', 'effective_technology_Rank_3',
    'effective_technology_Rank_4', 'effective_technology_Rank_5', 'effective_technology_Rank_6',
    'effective_technology_Rank_7'
]
effective_counts = df[rank_cols_tech].melt(value_name="Technology")["Technology"].value_counts()

df[rank_cols_tech] = df[rank_cols_tech].astype(str)
tech_summary = pd.DataFrame()

for i, col in enumerate(rank_cols_tech, 1):
    tech_summary = pd.concat([
        tech_summary,
        df[col].value_counts().rename(f"Rank_{i}")
    ], axis=1)

tech_summary = tech_summary.fillna(0).astype(int)
tech_summary["Total"] = tech_summary.sum(axis=1)
tech_summary = tech_summary.drop(index=["nan", "NaN", "None"], errors="ignore")
tech_summary = tech_summary.sort_values("Total", ascending=True)

# ========== VISIBILITY & CONTROL OBJECTIVES SUMMARY ==========
rank_cols_control = [
    'visibility_and_control_Rank_1', 'visibility_and_control_Rank_2', 'visibility_and_control_Rank_3',
    'visibility_and_control_Rank_4', 'visibility_and_control_Rank_5', 'visibility_and_control_Rank_6'
]
visibility_counts = df[rank_cols_control].melt(value_name="Control_Field")["Control_Field"].value_counts()
df[rank_cols_control] = df[rank_cols_control].astype(str)
control_summary = pd.DataFrame()

for i, col in enumerate(rank_cols_control, 1):
    control_summary = pd.concat([
        control_summary,
        df[col].value_counts().rename(f"Rank_{i}")
    ], axis=1)

control_summary = control_summary.fillna(0).astype(int)
control_summary["Total"] = control_summary.sum(axis=1)
control_summary = control_summary.drop(index=["nan", "NaN", "None"], errors="ignore")
control_summary = control_summary.sort_values("Total", ascending=True)

effective_counts_df = effective_counts.reset_index()
effective_counts_df.columns = ["Technology", "Frequency"]

visibility_counts_df = visibility_counts.reset_index()
visibility_counts_df.columns = ["Control_Field", "Frequency"]

print("=== Effective Technology Rank Summary ===")
print(tech_summary)

print("\n=== Visibility & Control Objective Rank Summary ===")
print(control_summary)

# Optional: save to CSV
tech_summary.to_csv("effective_technology_ranking_summary.csv")
control_summary.to_csv("visibility_control_summary.csv")
print("\nSummaries saved as CSV files.")

# ========== PLOT: EFFECTIVE TECHNOLOGIES ==========
plt.figure(figsize=(12, 6))
ax1 = sns.barplot(
    x=tech_summary["Total"],
    y=[label.replace('_', ' ').title() for label in tech_summary.index],
    palette="crest"
)
plt.title("Total Mentions of Effective Technologies Across All Ranks", fontsize=14)
plt.xlabel("Total Mentions", fontsize=12)
plt.ylabel("Technology", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# ========== PLOT: CONTROL OBJECTIVES ==========
plt.figure(figsize=(14, 8))
ax2 = sns.barplot(
    x=control_summary["Total"],
    y=[label.replace('_', ' ').title() for label in control_summary.index],
    palette="flare"
)
plt.title("Total Mentions of Visibility and Control Objectives Across All Ranks", fontsize=14)
plt.xlabel("Total Mentions", fontsize=12)
plt.ylabel("Control Objective", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()