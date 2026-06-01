#!/usr/bin/env python
# coding: utf-8

# # Community Dectection - Replicating Cho et al. (2023)
# ### County-to-county migration data
# ### Louvain method

# In[ ]:


import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
import geopandas as gpd
import igraph as ig

from collections import defaultdict
from itertools import combinations

try:
    display
except NameError:
    def display(value):
        if hasattr(value, "to_string"):
            print(value.to_string(index=False))
        else:
            print(value)


# # Link

# In[ ]:


# websites: 


# In[ ]:


colspecs = (
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (13, 43),
    (43, 78),
    (79, 109),
    (109, 144),
    (145, 153),
    (154, 162),
    (163, 171),
    (172, 180),
    (181, 189),
    (190, 198),
    (199, 207),
    (208, 216)
)
data_url = "Net_Gross_US.txt"

df = pd.read_fwf(data_url, colspecs=colspecs, header=None, encoding="latin1")


# In[ ]:


df.columns = (
    "state_code_A",
    "county_code_A",
    "state_code_B",
    "county_code_B",
    "state_A",
    "county_A",
    "state_B",
    "county_B",
    "flow_B_A",
    "flow_B_A_MOE",
    "flow_A_B",
    "flow_A_B_MOE",
    "net_flow_B_A",
    "net_flow_B_A_MOE",
    "gross_flow_A_B",
    "gross_flow_A_B_MOE"
)


# In[ ]:


link = df[["state_code_A","county_code_A","state_code_B","county_code_B", "state_A", "county_A", "state_B", "county_B","flow_B_A"]]


# In[ ]:


link.dtypes


# In[ ]:


link


# In[ ]:


link = link.dropna()


link["state_code_A"] = link["state_code_A"].astype(int).astype(str).str.zfill(2)
link["state_code_B"] = link["state_code_B"].astype(int).astype(str).str.zfill(2)

link["county_code_A"] = link["county_code_A"].astype(int).astype(str).str.zfill(3)
link["county_code_B"] = link["county_code_B"].astype(int).astype(str).str.zfill(3)

link["fips_A"] = link["state_code_A"] + link["county_code_A"]
link["fips_B"] = link["state_code_B"] + link["county_code_B"]


# In[ ]:


link2 = link[(link['flow_B_A'] > 500) & (link['fips_A'] != link['fips_B'])]


# In[ ]:


link2 = link[(link['flow_B_A'] > 50) & (link['fips_A'] != link['fips_B'])]


# # Node

# In[ ]:


node = pd.read_csv('R50010085_SL050.csv')


# In[ ]:


node2 = node[['Geo_FIPS','Geo_STATE','Geo_COUNTY','Geo_QName','SE_A00001_001', 'SE_A14028_001', 'SE_A10001_001']]


# In[ ]:


node2['GEOID'] = node2['Geo_FIPS'].astype(int).astype(str).str.zfill(5)


# In[ ]:


node2.columns = [
    "Geo_FIPS", "Geo_STATE", "Geo_County", "Geo_QName",
    "pop", "gini", "housing_units", "GEOID"
]


# In[ ]:


node2 = node2.copy()


# # Networkx

# In[ ]:


# set link
g = nx.from_pandas_edgelist(link2, source = 'fips_B', target = 'fips_A', edge_attr='flow_B_A')
#g.edges(data=True)


# In[ ]:


node_attr = node2.set_index('GEOID').to_dict('index')
nx.set_node_attributes(g, node_attr)


# In[ ]:


degree_dict = dict(g.degree(weight='flow_B_A'))
nx.set_node_attributes(g, degree_dict, 'degree')


# In[ ]:


largest_nodes = max(nx.connected_components(g), key=len)
g = g.subgraph(largest_nodes).copy()


# ## Louvain

# In[ ]:


louv = community_louvain.best_partition(g, weight = 'flow_B_A', resolution = 1.1) # dictionary format

print(community_louvain.modularity(louv, g, weight='flow_B_A'))
node2['louv'] = node2['GEOID'].map(louv) # save as a column in nodes dataframe
print(len(node2['louv'].unique()))
node2.head()


# In[ ]:


# Drop old resolution columns if this cell is rerun.
old_comm_cols = [c for c in node2.columns if str(c).startswith('comm_')]
node2 = node2.drop(columns=old_comm_cols)

res_analysis_records = []
comm_columns = {}

for i in range(1, 21):
    gamma = round(i * 0.05, 2)

    for run in range(10):
        partition = community_louvain.best_partition(
            g,
            weight='flow_B_A',
            resolution=gamma
        )

        comm_columns[f'comm_{i}_{run}'] = node2['GEOID'].map(partition)

        num_communities = len(set(partition.values()))
        modularity_score = community_louvain.modularity(
            partition,
            g,
            weight='flow_B_A'
        )

        res_analysis_records.append({
            'resolution': gamma,
            'run': run + 1,
            'num_communities': num_communities,
            'modularity_score': modularity_score
        })

comm_df = pd.DataFrame(comm_columns)
node2 = pd.concat([node2, comm_df], axis=1)
res_analysis_df = pd.DataFrame(res_analysis_records)
res_analysis_df.head()


# In[ ]:


import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(
    data=res_analysis_df,
    x='resolution',
    y='num_communities',
    ax=axes[0],
    color='skyblue'
)
axes[0].set_title('Resolution vs Number of Communities')
axes[0].set_xlabel('Resolution (gamma)')
axes[0].set_ylabel('Number of Communities')
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(
    data=res_analysis_df,
    x='resolution',
    y='modularity_score',
    ax=axes[1],
    color='lightgreen'
)
axes[1].set_title('Resolution vs Modularity Score')
axes[1].set_xlabel('Resolution (gamma)')
axes[1].set_ylabel('Modularity Score')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

res_analysis_df.head()


# In[ ]:


# cooccrence edge list
node2 = node2.loc[:, ~node2.columns.duplicated()].copy()
comm_cols = [c for c in node2.columns if str(c).startswith('comm_')]
pair_counts = defaultdict(int)

for col in comm_cols:
    for i, group in node2.groupby(col).groups.items():
        node_list = node2.loc[group, 'GEOID'].tolist()
        for u, v in combinations(sorted(node_list), 2):
            pair_counts[(u, v)] += 1

# edge list DataFrame
cooccurence = (
    pd.Series(pair_counts)
      .reset_index()
      .rename(columns={'level_0':'source', 'level_1':'target', 0:'same_count'})
)

# normalize the value
cooccurence['weight'] = cooccurence['same_count'] / len(comm_cols)
cooccurence


# In[ ]:


# Create a network based on the cooccurrence edge list
# apply louvain algorithm to the network
g_cooc = nx.from_pandas_edgelist(
    cooccurence,
    source='source',
    target='target',
    edge_attr='weight'
)

g_cooc.add_nodes_from(node2['GEOID'])

consensus_louv = community_louvain.best_partition(g_cooc, weight='weight')
node2['consensus_comm'] = node2['GEOID'].map(consensus_louv)

print(f"Final number of consensus communities: {len(set(consensus_louv.values()))}")
node2[['GEOID', 'Geo_QName', 'consensus_comm']].head()


# # Multiresolution membership inconsistency analysis

# In[ ]:


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cut_tree
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import re
import sys
sys.setrecursionlimit(100000)


# In[ ]:


# Counties are the nodes in this analysis.
# Migration flows are weighted edges between county FIPS/GEOID values.


# In[ ]:


help(dendrogram)


# The hierarchical decomposition below uses county-level community membership profiles across resolution settings. The dendrogram is truncated because the full county-level hierarchy contains more than 3,000 counties.

# In[ ]:


# (1) build county-level community membership profiles
comm_cols = [c for c in node2.columns if str(c).startswith('comm_')]
print('Community assignment columns used:')
print(comm_cols)
print(f'Number of community assignment columns used: {len(comm_cols)}')

profile_df = node2[['GEOID'] + comm_cols].drop_duplicates('GEOID').copy()
profile_df['GEOID'] = profile_df['GEOID'].astype(str).str.zfill(5)

# Remove counties with missing values in all community assignment columns.
profile_df = profile_df.dropna(subset=comm_cols, how='all').copy()
county_ids = profile_df['GEOID'].tolist()
print(f'Counties used in membership profile clustering: {len(profile_df)}')

# Community IDs are categorical, so use one-hot profile features instead of numeric IDs.
X = pd.get_dummies(profile_df[comm_cols].astype(str), prefix=comm_cols)
print(f'Membership profile feature matrix shape: {X.shape}')

# Cosine distance compares county community-membership patterns across resolutions.
D_profile = squareform(pdist(X.to_numpy(dtype=np.float32), metric='cosine'))
D_profile = np.nan_to_num(D_profile, nan=1.0)
D_profile = np.clip(D_profile, 0, 1)
np.fill_diagonal(D_profile, 0)

profile_off_diag = D_profile[~np.eye(len(D_profile), dtype=bool)]
print('Membership profile distance summary:')
print(pd.Series(profile_off_diag).describe()[['min', 'mean', '50%', 'max']].rename({'50%': 'median'}))

# Average linkage hierarchy based on membership-profile distance.
Z_profile = linkage(squareform(D_profile, checks=False), method='average')
Z = Z_profile  # keep Z available for any later exploratory cells

fig, ax = plt.subplots(figsize=(10, 6))
dendrogram(
    Z_profile,
    truncate_mode='lastp',
    p=50,
    no_labels=True,
    show_leaf_counts=True,
    distance_sort='descending',
    ax=ax
)
ax.set_xlabel('County groups (truncated display)')
ax.set_ylabel('Community membership profile distance')
ax.set_title('County-level Hierarchical Decomposition')
plt.tight_layout()
plt.show()


# In[ ]:


# Create county-level hierarchical clusters from membership-profile distances.
dend_cluster_count = 12


def fit_profile_clusters(n_clusters):
    try:
        from sklearn.cluster import AgglomerativeClustering

        try:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='average'
            )

        labels = model.fit_predict(D_profile) + 1
        method = 'sklearn AgglomerativeClustering'
    except ModuleNotFoundError:
        print('sklearn is not installed in this environment; using scipy cut_tree fallback with average linkage.')
        labels = cut_tree(Z_profile, n_clusters=[n_clusters]).flatten() + 1
        method = 'scipy cut_tree fallback'

    labels = labels.astype(int)
    sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    largest_pct = sizes.iloc[0] / len(labels)
    return labels, sizes, largest_pct, method

labels_12, sizes_12, largest_pct_12, cluster_method = fit_profile_clusters(dend_cluster_count)
selected_cluster_count = dend_cluster_count
selected_labels = labels_12
selected_sizes = sizes_12
selected_largest_pct = largest_pct_12

print(f'Hierarchical clustering method: {cluster_method}')
print('Dendrogram 12-cluster unique count:', pd.Series(labels_12).nunique())
print('Top 12 cluster sizes for 12-cluster solution:')
display(sizes_12.rename_axis('dend_cluster_12').reset_index(name='county_count').head(12))
print(f'Largest cluster percentage for 12-cluster solution: {largest_pct_12:.2%}')

if largest_pct_12 > 0.80:
    print('The 12-cluster solution is highly imbalanced.')
    candidate_results = []
    for k in [8, 12, 15]:
        labels_k, sizes_k, largest_pct_k, _ = fit_profile_clusters(k)
        candidate_results.append({
            'n_clusters': k,    
            'unique_clusters': pd.Series(labels_k).nunique(),
            'largest_cluster_pct': largest_pct_k
        })
        if largest_pct_k < selected_largest_pct:
            selected_cluster_count = k
            selected_labels = labels_k
            selected_sizes = sizes_k
            selected_largest_pct = largest_pct_k

    candidate_summary = pd.DataFrame(candidate_results)
    print('Alternative cluster balance check:')
    display(candidate_summary)

profile_cluster_df = pd.DataFrame({
    'GEOID': profile_df['GEOID'].astype(str).str.zfill(5).values,
    'dend_cluster_12': selected_labels.astype(int)
})

unique_dend_clusters = profile_cluster_df['dend_cluster_12'].nunique()
print(f'Selected hierarchical cluster count: {selected_cluster_count}')
print('Final dend_cluster_12 unique count:', unique_dend_clusters)
print('Top 12 largest final hierarchical clusters by county count:')
display(
    profile_cluster_df['dend_cluster_12']
    .value_counts(dropna=True)
    .rename_axis('dend_cluster_12')
    .reset_index(name='county_count')
    .head(12)
)
print(f'Largest final cluster percentage: {selected_largest_pct:.2%}')

if selected_cluster_count == dend_cluster_count and unique_dend_clusters != dend_cluster_count:
    raise ValueError(f'Expected {dend_cluster_count} dendrogram clusters, but got {unique_dend_clusters}.')

# merge county-level profile clusters back to node2 for later maps
node2['GEOID'] = node2['GEOID'].astype(str).str.zfill(5)
old_dend_cols = [c for c in node2.columns if str(c).startswith('dend_')]
node2 = node2.drop(columns=old_dend_cols)
node2 = node2.merge(profile_cluster_df, on='GEOID', how='left')


# In[ ]:


# add comments to the following codes
# (3) Membership inconsistency (MeI)

def parse_gamma_from_col(colname: str, default_gamma=1.0):
    """
    extract columns from the same gamma (i.e., resolution)
    """
    nums = re.findall(r'\d+(?:\.\d+)?', str(colname))
    if nums:
        return float(nums[0])
    return float(default_gamma)


comm_cols = [c for c in node2.columns if str(c).startswith('comm')]
nodes = list(pd.Index(node2['GEOID']).astype(object))
node_index = pd.Index(nodes)

by_gamma = defaultdict(list)
for c in comm_cols:
    gamma = parse_gamma_from_col(c)
    labels_raw = node2.set_index('GEOID').reindex(node_index)[c]
    codes = pd.Categorical(labels_raw.astype(object)).codes  
    if np.any(codes < 0):
        fill_vals = [f'NA_{i}' if codes[i] < 0 else labels_raw.iloc[i] for i in range(len(codes))]
        codes = pd.Categorical(pd.Series(fill_vals)).codes
    by_gamma[gamma].append(np.asarray(codes, dtype=int))

partitions_by_gamma = dict(sorted(by_gamma.items(), key=lambda kv: kv[0]))


def compute_mei(partitions_by_gamma: dict, max_pairs_per_gamma: int | None = 50000, random_state: int = 0) -> pd.Series:
    rng = np.random.default_rng(random_state)
    gammas = sorted(partitions_by_gamma.keys())
    any_gamma = gammas[0]
    N = len(partitions_by_gamma[any_gamma][0])

    jacc_by_gamma = np.zeros((len(gammas), N), dtype=float)

    for g_idx, gamma in enumerate(gammas):
        configs = partitions_by_gamma[gamma]
        R = len(configs)
        partner_sets = [[None]*R for _ in range(N)]
        for r, labels in enumerate(configs):
            comm2members = defaultdict(list)
            for i, c in enumerate(labels):
                comm2members[c].append(i)
            comm2members = {k: set(v) for k, v in comm2members.items()}
            for i, c in enumerate(labels):
                partner_sets[i][r] = comm2members[c]

        all_pairs = list(combinations(range(R), 2))
        if (max_pairs_per_gamma is not None) and (len(all_pairs) > max_pairs_per_gamma):
            all_pairs = list(rng.choice(len(all_pairs), size=max_pairs_per_gamma, replace=False))
            all_pairs = [list(combinations(range(R), 2))[k] for k in all_pairs]

        for i in range(N):
            s = 0.0; cnt = 0
            for a, b in all_pairs:
                A = partner_sets[i][a]; B = partner_sets[i][b]
                inter = len(A & B); union = len(A | B)
                j = inter/union if union>0 else 1.0
                s += j; cnt += 1
            jacc_by_gamma[g_idx, i] = (s/cnt) if cnt>0 else 1.0

    jacc_mean = jacc_by_gamma.mean(axis=0)
    eps = 1e-12
    mei = 1.0 / np.maximum(jacc_mean, eps)
    return pd.Series(mei, name='MeI')

mei_series = compute_mei(partitions_by_gamma, max_pairs_per_gamma=50000, random_state=0)
mei_series.index = nodes 
mei_series.sort_values(ascending=False).head(10)


# In[ ]:


old_mei_cols = [c for c in node2.columns if str(c).startswith('MeI')]
node2 = node2.drop(columns=old_mei_cols)

mei_df = pd.DataFrame({
    'GEOID': list(nodes),
    'MeI':   mei_series.values
})
node2 = node2.merge(mei_df, on='GEOID', how='left')


# In[ ]:


# Create a bar graph like Figure 5. in Cho et al. (2023)
# County-level summary: show counties with the highest MeI values.
county_mei_bar = (
    node2[['GEOID', 'Geo_QName', 'MeI']]
    .dropna()
    .sort_values('MeI', ascending=False)
    .head(15)
    .sort_values('MeI')
)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(county_mei_bar['Geo_QName'], county_mei_bar['MeI'], color='steelblue')
ax.set_xlabel('Membership inconsistency (MeI)')
ax.set_ylabel('County')
ax.set_title('Counties with Highest Membership Inconsistency')
plt.tight_layout()
plt.show()


# # Mapping county-level communities
# 
# Counties are the unit of analysis in the final maps. County FIPS/GEOID values identify graph nodes, and county-to-county migration flows are weighted edges. State boundaries are used only as thin visual reference lines to make the county maps easier to read.

# In[ ]:


county = gpd.read_file('COUNTY_2019_US_SL050_2019-11-13_15-15-56-579/COUNTY_2019_US_SL050_Coast_Clipped.shp')
print('node2 columns:', list(node2.columns))
print('link2 columns:', list(link2.columns))
print('county columns:', list(county.columns))
print(county.dtypes)
county.head()


# In[ ]:


county["STATEFP"] = county["STATEFP"].astype(str).str.zfill(2)

# contiguous US only
exclude_fips = {"02",  # Alaska
                "15",  # Hawaii
                "60",  # American Samoa
                "66",  # Guam
                "69",  # Northern Mariana Islands
                "72",  # Puerto Rico
                "78"}  # U.S. Virgin Islands

county_conus = county[~county["STATEFP"].isin(exclude_fips)].copy()
state_boundaries = county_conus.dissolve(by='STATEFP', as_index=False)

# check if county boundaries look good
# State boundaries are only an overlay/background for readability.
ax = county_conus.plot(figsize=(12, 7), linewidth=0.1, edgecolor="white", facecolor="#e0e0e0")
ax.set_title('County Boundaries (Contiguous U.S.)')
ax.set_axis_off()
plt.tight_layout()
plt.show()


# Colors represent different county-level consensus communities. The full legend is omitted because the number of communities is large.

# In[ ]:


county_conus['GEOID'] = county_conus['GEOID'].astype(str).str.zfill(5)
node2['GEOID'] = node2['GEOID'].astype(str).str.zfill(5)

map_cols = ['GEOID', 'consensus_comm']
for col in ['comm_1_0', 'comm_20_0', 'MeI', 'dend_cluster_12']:
    if col in node2.columns:
        map_cols.append(col)

county_merged = county_conus.merge(
    node2[map_cols].drop_duplicates('GEOID'),
    on='GEOID',
    how='left'
)

missing_count = county_merged['consensus_comm'].isna().sum()
merged_count = county_merged['consensus_comm'].notna().sum()
print(f"County nodes in graph: {g.number_of_nodes()}")
print(f"County polygons in final plotting GeoDataFrame: {len(county_merged)}")
print(f"Counties successfully merged with shapefile: {merged_count} of {len(county_merged)}")
print(f"Counties with missing community labels: {missing_count}")
if 'comm_1_0' in county_merged.columns:
    print(f"Lower resolution county communities: {county_merged['comm_1_0'].nunique(dropna=True)}")
if 'comm_20_0' in county_merged.columns:
    print(f"Higher resolution county communities: {county_merged['comm_20_0'].nunique(dropna=True)}")
print(f"Consensus county communities: {county_merged['consensus_comm'].nunique(dropna=True)}")
if 'dend_cluster_12' in county_merged.columns:
    print(f"Dendrogram clusters after 12-cluster cut: {county_merged['dend_cluster_12'].nunique(dropna=True)}")
print("No state-level aggregation was introduced; final maps below use county polygons.")

consensus_summary = (
    county_merged['consensus_comm']
    .value_counts(dropna=True)
    .rename_axis('consensus_comm')
    .reset_index(name='county_count')
    .head(10)
)
print("Top 10 largest consensus communities by county count:")
display(consensus_summary)

plot_data = county_merged.copy()
plot_data['consensus_comm'] = plot_data['consensus_comm'].astype('Int64').astype('category')

fig, ax = plt.subplots(figsize=(12, 7))
plot_data.plot(
    column='consensus_comm',
    cmap='tab20',
    linewidth=0.05,
    edgecolor='white',
    categorical=True,
    legend=False,
    missing_kwds={'color': 'lightgray', 'label': 'No data'},
    ax=ax
)
state_boundaries.boundary.plot(ax=ax, linewidth=0.4, color='gray')

ax.set_title('County-level Majority Consensus Communities')
ax.set_axis_off()
plt.tight_layout()
plt.show()

county_merged[['GEOID', 'consensus_comm']].head()


# ## Required county-level community maps
# 

# These maps compare how community structure changes across resolution values. The focus is on spatial fragmentation and regional pattern, not on identifying every community ID.

# In[ ]:


# Lower resolution vs higher resolution community maps at the county level.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, col, title in zip(
    axes,
    ['comm_1_0', 'comm_20_0'],
    ['County-level Lower Resolution Communities (gamma = 0.05)', 'County-level Higher Resolution Communities (gamma = 1.00)']
):
    plot_data = county_merged.copy()
    plot_data[col] = plot_data[col].astype('Int64').astype('category')
    print(f"{title}: {plot_data[col].nunique(dropna=True)} unique communities")
    plot_data.plot(
        column=col,
        cmap='tab20',
        linewidth=0.05,
        edgecolor='white',
        categorical=True,
        legend=False,
        missing_kwds={'color': 'lightgray', 'label': 'No data'},
        ax=ax
    )
    state_boundaries.boundary.plot(ax=ax, linewidth=0.4, color='gray')
    ax.set_title(title)
    ax.set_axis_off()

plt.tight_layout()
plt.show()


# In[ ]:


# County-level MeI map in a sequential color scheme.
fig, ax = plt.subplots(figsize=(12, 7))
county_merged.plot(
    column='MeI',
    cmap='viridis',
    linewidth=0.05,
    edgecolor='white',
    legend=True,
    legend_kwds={'label': 'Membership Inconsistency (MeI)'},
    missing_kwds={'color': 'lightgray', 'label': 'No data'},
    ax=ax
)
state_boundaries.boundary.plot(ax=ax, linewidth=0.4, color='gray')

ax.set_title('County-level Membership Inconsistency (MeI)')
ax.set_axis_off()
plt.tight_layout()
plt.show()


# This map groups counties into 12 hierarchical clusters based on their community membership profiles across resolution settings. Counties remain the unit of analysis, and state boundaries are shown only as geographic reference lines.

# In[ ]:


# Map county-level hierarchical clusters from membership profiles.
dend_col = 'dend_cluster_12'
dend_plot = county_merged.copy()
dend_plot[dend_col] = dend_plot[dend_col].astype('Int64').astype('category')

non_missing_dend = dend_plot[dend_col].notna().sum()
unique_dend_final = dend_plot[dend_col].nunique(dropna=True)
print(f'County polygons in dendrogram plotting GeoDataFrame: {len(dend_plot)}')
print(f'Non-missing dend_cluster_12 labels: {non_missing_dend}')
print(f'Unique dend_cluster_12 clusters in final map: {unique_dend_final}')

if unique_dend_final != 12:
    print(f'Final map has {unique_dend_final} clusters because the balance check selected a different cluster count.')

dend_summary = (
    county_merged[dend_col]
    .value_counts(dropna=True)
    .rename_axis(dend_col)
    .reset_index(name='county_count')
    .head(12)
)
print('Top hierarchical clusters by county count:')
display(dend_summary)

fig, ax = plt.subplots(figsize=(12, 7))
dend_plot.plot(
    column=dend_col,
    cmap='tab20',
    linewidth=0.05,
    edgecolor='white',
    categorical=True,
    legend=True,
    legend_kwds={
        'title': 'Hierarchical Cluster',
        'loc': 'center left',
        'bbox_to_anchor': (1.02, 0.5)
    },
    missing_kwds={'color': 'lightgray', 'label': 'No data'},
    ax=ax
)
state_boundaries.boundary.plot(ax=ax, linewidth=0.4, color='gray')

ax.set_title('County-level Hierarchical Clusters Based on Membership Profiles')
ax.set_axis_off()
plt.tight_layout()
plt.show()
