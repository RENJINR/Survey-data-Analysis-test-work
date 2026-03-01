# Generate a full Python script string containing all steps from clustering to scenario modeling
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from sklearn.metrics import classification_report
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

# Load data
df = pd.read_csv(r"D:\RENJIN RAJU\MASTERS\SEMESTER 3\THESIS\Research\Analysis\Survey data cleaned.csv", encoding="ISO-8859-1")

# ========== CLUSTER ANALYSIS ==========
cluster_features = [
    'Disruption_Natural_Disasters', 'Disruption_Supplier_Failures',
    'Disruption_Cybersecurity_Threats', 'Disruption_Global_Crises',
    'Disruption_Transportation_Delays', 'adoption_Artificial_Intelligence_AI',
    'adoption_Internet_of_Things_IoT', 'adoption_Blockchain', 'adoption_Cloud_Computing',
    'adoption_Robotic_and_Automation', 'adoption_Big_Data_Analytics', 'adoption_Simulation',
    'implemented_AI', 'implemented_IoT', 'implemented_Blockchain', 'implemented_Cloud_Computing',
    'implemented_Robotics_and_Automation', 'implemented_Big_Data_Analytics',
    'barriers_High_Implementation_Costs', 'barriers_Lack_of_Skilled_Workforce',
    'barriers_Integration_Issues', 'barriers_Resistance_to_Change'
]

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(cluster_scaled)

pca = PCA(n_components=2)
components = pca.fit_transform(cluster_scaled)
df['PCA1'], df['PCA2'] = components[:, 0], components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100)
plt.title("K-Means Clustering of Respondents")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.tight_layout()
plt.show()

# ========== DECISION TREE MODELING ==========
features = [
    'Disruption_Global_Crises', 'Disruption_Supplier_Failures',
    'Disruption_Cybersecurity_Threats', 'adoption_Artificial_Intelligence_AI',
    'adoption_Blockchain', 'barriers_High_Implementation_Costs', 'barriers_Resistance_to_Change'
]
df['Cloud_Implemented_Binary'] = df['implemented_Cloud_Computing'].apply(lambda x: 1 if x > 0 else 0)

X = df[features]
y = df['Cloud_Implemented_Binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(14, 7))
plot_tree(tree, feature_names=features, class_names=["Not Implemented", "Implemented"],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree for Cloud Computing Implementation")
plt.tight_layout()
plt.show()

# ========== SCENARIO MODELING: AI Implementation ==========
scenario_df = df.copy()
scenario_df['adoption_Artificial_Intelligence_AI'] = 1
scenario_df['implemented_AI'] = 5
scenario_df['predicted_cloud_prob'] = tree.predict_proba(scenario_df[features])[:, 1]
scenario_df['prediction_shift'] = scenario_df['predicted_cloud_prob'] - df['Cloud_Implemented_Binary']

print("\\nScenario: All Implement AI")
print("Original Cloud Adoption Rate:", df['Cloud_Implemented_Binary'].mean())
print("Predicted Rate:", scenario_df['predicted_cloud_prob'].mean())
print("Average Shift in Probability:", scenario_df['prediction_shift'].mean())

# ========== SCENARIO MODELING: Full Tech Implementation ==========
tech_implementation_features = [
    'implemented_AI', 'implemented_IoT', 'implemented_Blockchain',
    'implemented_Cloud_Computing', 'implemented_Robotics_and_Automation',
    'implemented_Big_Data_Analytics'
]
scenario_all = df.copy()
for col in tech_implementation_features:
    scenario_all[col] = 5

scenario_all['predicted_cloud_prob_all_techs'] = tree.predict_proba(scenario_all[features])[:, 1]
scenario_all['prediction_shift_all_techs'] = scenario_all['predicted_cloud_prob_all_techs'] - df['Cloud_Implemented_Binary']

print("\\nScenario: All Technologies Implemented")
print("Original Cloud Adoption Rate:", df['Cloud_Implemented_Binary'].mean())
print("Predicted Rate:", scenario_all['predicted_cloud_prob_all_techs'].mean())
print("Average Shift in Probability:", scenario_all['prediction_shift_all_techs'].mean())

df['Cloud_Implemented_Binary'] = df['implemented_Cloud_Computing'].apply(lambda x: 1 if x > 0 else 0)

# ========= 1. LOGISTIC REGRESSION ==========
features = [
    'Disruption_Global_Crises', 'Disruption_Supplier_Failures',
    'Disruption_Cybersecurity_Threats', 'adoption_Artificial_Intelligence_AI',
    'adoption_Blockchain', 'barriers_High_Implementation_Costs',
    'barriers_Resistance_to_Change'
]
X = df[features]
y = df['Cloud_Implemented_Binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
print("\\nLOGISTIC REGRESSION RESULTS")
print(classification_report(y_test, y_pred))

# ========= 2. ASSOCIATION RULE MINING ==========
# Select binary adoption columns
adoption_cols = [
    'adoption_Artificial_Intelligence_AI', 'adoption_Internet_of_Things_IoT',
    'adoption_Blockchain', 'adoption_Cloud_Computing',
    'adoption_Robotic_and_Automation', 'adoption_Big_Data_Analytics',
    'adoption_Simulation'
]
df_apriori = df[adoption_cols].copy()
df_apriori = df_apriori.applymap(lambda x: True if x > 0 else False)

frequent_items = apriori(df_apriori, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)
rules = rules.sort_values(by="confidence", ascending=False)
print("\\nTOP ASSOCIATION RULES:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15))

# ========= 3. PROPENSITY SCORE MATCHING: MULTIPLE TECHS ==========
print("\\nPROPENSITY SCORE MATCHING RESULTS (ATE)")
psm_features = ['Disruption_Global_Crises', 'Disruption_Cybersecurity_Threats',
                'barriers_High_Implementation_Costs', 'barriers_Resistance_to_Change']
outcome = 'Cloud_Implemented_Binary'

implemented_cols = [
    'implemented_AI', 'implemented_IoT', 'implemented_Blockchain',
    'implemented_Cloud_Computing', 'implemented_Robotics_and_Automation',
    'implemented_Big_Data_Analytics'
]

for tech in implemented_cols:
    label = tech.split('_', 1)[-1]
    df['Treatment'] = df[tech].apply(lambda x: 1 if x > 0 else 0)

    X_psm = df[psm_features]
    y_psm = df['Treatment']

    model = LogisticRegression()
    model.fit(X_psm, y_psm)
    df['propensity_score'] = model.predict_proba(X_psm)[:, 1]

    treated = df[df['Treatment'] == 1]
    control = df[df['Treatment'] == 0]

    if treated.empty or control.empty:
        print(f"{label}: Not enough data to run PSM.")
        continue

    neighbors = NearestNeighbors(n_neighbors=1).fit(control[['propensity_score']])
    distances, indices = neighbors.kneighbors(treated[['propensity_score']])
    matched_control = control.iloc[indices.flatten()]

    treated_outcome = treated[outcome].reset_index(drop=True)
    control_outcome = matched_control[outcome].reset_index(drop=True)

    if not control_outcome.empty:
        ate = treated_outcome.mean() - control_outcome.mean()
        print(f"{label}: ATE = {ate:.3f}")
    else:
        print(f"{label}: Matching failed.")

# ========= 4. BAYESIAN NETWORK MODELING ==========
bayes_df = df[[
    'Cloud_Implemented_Binary', 'adoption_Artificial_Intelligence_AI',
    'adoption_Internet_of_Things_IoT', 'adoption_Blockchain',
    'barriers_Resistance_to_Change', 'Disruption_Cybersecurity_Threats'
]].copy().astype(int)

model = BayesianNetwork([
    ('adoption_Artificial_Intelligence_AI', 'Cloud_Implemented_Binary'),
    ('adoption_Blockchain', 'Cloud_Implemented_Binary'),
    ('barriers_Resistance_to_Change', 'Cloud_Implemented_Binary'),
    ('Disruption_Cybersecurity_Threats', 'Cloud_Implemented_Binary')
])
model.fit(bayes_df, estimator=MaximumLikelihoodEstimator)
print("\\nBAYESIAN NETWORK MODEL FIT COMPLETE")
# Print Conditional Probability Distributions
for cpd in model.get_cpds():
    print(f"\\n{cpd}")
