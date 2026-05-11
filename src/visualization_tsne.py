import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

# 1. Load the cleaned dataset
df = pd.read_csv('cleaned_environmental_data_10K.csv')

print("1. Extracting 3-mers and calculating TF-IDF...")
def get_kmers(sequence, k=3):
    if not isinstance(sequence, str):
        return []
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Convert sequences to TF-IDF matrix
vectorizer = TfidfVectorizer(analyzer=get_kmers)
X_sequence = vectorizer.fit_transform(df['Sequence'])

print("2. Optimizing Dimensions with TruncatedSVD...")
# Best practice: Reduce the thousands of 3-mer dimensions down to 50 before t-SNE
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_sequence)

print("3. Running t-SNE Dimensionality Reduction (This may take a minute)...")
# Perplexity controls the balance between local and global aspects of your data (usually 5 to 50)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
tsne_embedding = tsne.fit_transform(X_svd)

# Add coordinates back to dataframe
df['tSNE_1'] = tsne_embedding[:, 0]
df['tSNE_2'] = tsne_embedding[:, 1]

print("4. Calculating Annual Aggregates...")
tavg_cols = [f'tavg_{str(i).zfill(2)}' for i in range(1, 13)]
prec_cols = [f'prec_{str(i).zfill(2)}' for i in range(1, 13)]
srad_cols = [f'srad_{str(i).zfill(2)}' for i in range(1, 13)]

# Calculate row-wise means/sums safely
df['Mean_Annual_Temp'] = df[tavg_cols].mean(axis=1)
df['Total_Annual_Prec'] = df[prec_cols].sum(axis=1)
df['Mean_Annual_Solar_Rad'] = df[srad_cols].mean(axis=1)

print("5. Generating Visualizations...")
# Create a 2x3 grid to hold our 5 plots
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('Protein Sequence Space (t-SNE) Colored by Environmental Factors', fontsize=22, y=0.98)

scatter_kws = {'s': 15, 'alpha': 0.8, 'edgecolor': 'none'}

# Plot 1: NASA Collection Day Temp
sns.scatterplot(
    data=df, x='tSNE_1', y='tSNE_2', 
    hue='NASA_Temp_C', palette='coolwarm', 
    ax=axes[0, 0], **scatter_kws
)
axes[0, 0].set_title('NASA Surface Temp (°C)', fontsize=16)

# Plot 2: Mean Annual Temp
sns.scatterplot(
    data=df, x='tSNE_1', y='tSNE_2', 
    hue='Mean_Annual_Temp', palette='inferno', 
    ax=axes[0, 1], **scatter_kws
)
axes[0, 1].set_title('Mean Annual Temp (WorldClim)', fontsize=16)

# Plot 3: Total Annual Precipitation
sns.scatterplot(
    data=df, x='tSNE_1', y='tSNE_2', 
    hue='Total_Annual_Prec', palette='viridis', 
    ax=axes[0, 2], **scatter_kws
)
axes[0, 2].set_title('Total Annual Precipitation', fontsize=16)

# Plot 4: Mean Annual Solar Radiation
sns.scatterplot(
    data=df, x='tSNE_1', y='tSNE_2', 
    hue='Mean_Annual_Solar_Rad', palette='magma', 
    ax=axes[1, 0], **scatter_kws
)
axes[1, 0].set_title('Mean Annual Solar Radiation', fontsize=16)

# Plot 5: NASA Collection Day Radiation
sns.scatterplot(
    data=df, x='tSNE_1', y='tSNE_2', 
    hue='NASA_Radiation', palette='plasma', 
    ax=axes[1, 1], **scatter_kws
)
axes[1, 1].set_title('NASA Collection Day Radiation', fontsize=16)

# Hide the 6th empty subplot
axes[1, 2].axis('off')

# Clean up layout
for i, ax in enumerate(axes.flat):
    if i == 5: # Skip the hidden plot
        continue
    legend = ax.get_legend()
    if legend is not None:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("environmental_sequence_tsne_5plots.png", dpi=300, bbox_inches='tight')
plt.show()