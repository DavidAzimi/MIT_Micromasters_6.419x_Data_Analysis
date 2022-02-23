import pandas as pd
import networkx as nx
import pickle
from scipy.sparse import csr_matrix
from scipy import stats as st
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

### Helper Functions
def density(G):
  n = G.number_of_nodes()
  m = G.number_of_edges()
  return float(2*m/(n*(n-1)))

def ave_len(lst):
  lengths = [len(i) for i in lst]
  return 0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths))

# Function by Pasindu Tennage
# https://stackoverflow.com/questions/51740306/scipy-kstest-typeerror-parse-args-takes-from-3-to-5-positional-arguments-but
def get_best_distribution(data):
  dist_names = ["powerlaw", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
  dist_results = []
  params = {}
  for dist_name in dist_names:
    dist = getattr(st, dist_name)
    param = dist.fit(data)
    params[dist_name] = param
    # Applying the Kolmogorov-Smirnov test
    D, p = st.kstest(data, dist_name, args=param)
    print("p value for "+dist_name+" = "+str(p))
    dist_results.append((dist_name, p))
  # select the best fitted distribution
  best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
  # store the name of the best fit and its p value
  print("Best fitting distribution: "+str(best_dist))
  print("Best p value: "+ str(p))
  print("Parameters for the best fit: "+ str(params[best_dist]))
  return best_dist, best_p, params[best_dist]

### Load Data
df_raw = pd.read_csv("./Cooffending/Cooffending.csv")

### Clean Data
df = df_raw.drop_duplicates(subset=['OffenderIdentifier','CrimeIdentifier'])
crimes = df.drop_duplicates(subset=['CrimeIdentifier'])

### Dictionaries
crimeLoc = pickle.load(open("./Cooffending/crimeLoc", "rb"))
crimeType = pickle.load(open("./Cooffending/crimeType", "rb"))
offender_ary = df.loc[:, 'OffenderIdentifier'].unique()
offenders = {} 
for i in range(len(offender_ary)):
  offenders[offender_ary[i]]=i

crime_ary = df.loc[:, 'CrimeIdentifier'].unique()
crimes = {}
for i in range(len(crime_ary)):
  crimes[crime_ary[i]]=i

### Feature Engineering
df.loc[:, 'TotalOffenders'] = df.loc[:, 'NumberYouthOffenders'] + df.loc[:, 'NumberAdultOffenders']  
df.loc[:, 'OffenderIdentifier'] = df.loc[:, 'OffenderIdentifier'].map(offenders)
df.loc[:, 'CrimeIdentifier'] = df.loc[:, 'CrimeIdentifier'].map(crimes)

### Visualize
# print(df.head())
# for col in df.columns:
#     print(df[col].value_counts().head(30))

# Top 5 crimes by number of offenders
# df.sort_values(by=['TotalOffenders'], ascending=False).drop_duplicates(subset=['CrimeIdentifier']).head(5).loc[:,('CrimeIdentifier', 'Municipality', 'TotalOffenders')].to_html('top5byNum.html')

### Co-offender Network
row = df.OffenderIdentifier
col = df.CrimeIdentifier
crime_mat = csr_matrix((np.ones(len(row)), (row, col)), shape=(row.max() + 1, col.max() + 1))
cooffender_mat = crime_mat @ crime_mat.T
cooffender_mat[cooffender_mat > 0] = 1 # unweighted
cooffender_mat.setdiag(0) # remove self-loops
cooffender_mat.eliminate_zeros() # avoids self loops w/ above since setdiag(0) does not itself change the sparsity pattern

### Graph the co-offender network
G = nx.convert_matrix.from_scipy_sparse_matrix(cooffender_mat)
# nx.draw(G, pos=nx.drawing.nx_agraph.graphviz_layout(G), with_labels=True) 
#print("Network G with isolates")
#G.number_of_nodes()
#nx.number_of_isolates(G)
#G.number_of_edges()
G.remove_nodes_from(list(nx.isolates(G))) # remove isolated nodes
n = G.number_of_nodes()
e = G.number_of_edges()
ave_d = 2*e / n
degs = sorted([d for n, d in G.degree()], reverse=True)
# print(ave_d, np.mean(degs))
degs = np.array(degs)
# len(degs[degs >= 100])
comps = sorted(nx.connected_components(G), key=len, reverse=True)
# len(comps)
LCC = max(nx.connected_components(G), key=len)

### Plot of Degree Distribution
# fig, ax = plt.subplots()
# ax.set_xlabel('Degree of node')
# ax.set_title('Degree Distribution of Co-offender Network (n=121159)')
# sns.histplot(degs, ax=ax, discrete=True)
# plt.show()

### Kolmogorov-Smirnov test of power=law versus poisson and other dists
#b,p,params = get_best_distribution(degs)

### Subgraphs
# Repeating co-offending 
cooffender_mat_r = crime_mat @ crime_mat.T
cooffender_mat_r[cooffender_mat_r == 1] = 0 # cooffend at least twice
cooffender_mat_r[cooffender_mat_r > 1] = 1 # unweighted 
cooffender_mat_r.setdiag(0) # remove self-loops
cooffender_mat_r.eliminate_zeros() # avoids self loops w/ above since setdiag(0) does not itself change the sparsity pattern
G_r = nx.convert_matrix.from_scipy_sparse_matrix(cooffender_mat_r)
G_r.remove_nodes_from(list(nx.isolates(G_r))) # remove isolated nodes

# Non-repeating co-offending
cooffender_mat_nr = crime_mat @ crime_mat.T
cooffender_mat_nr[cooffender_mat_nr > 1] = 0 # only keep if cooffended exactly once 
cooffender_mat_nr.setdiag(0) # remove self-loops
cooffender_mat_nr.eliminate_zeros() # avoids self loops w/ above since setdiag(0) does not itself change the sparsity pattern
G_nr = nx.convert_matrix.from_scipy_sparse_matrix(cooffender_mat_nr)
G_nr.remove_nodes_from(list(nx.isolates(G_nr))) # remove isolated nodes

### Compare Subgraphs
n_nr, n_r = G_nr.number_of_nodes(), G_r.number_of_nodes()
e_nr, e_r = G_nr.number_of_edges(), G_r.number_of_edges()
comps_nr = sorted(nx.connected_components(G_nr), key=len, reverse=True)
comps_r = sorted(nx.connected_components(G_r), key=len, reverse=True)
# ave_len(comps_r), ave_len(comps_nr), ave_len(comps)
# 
# 
### Comparing Largest Components
LCC = G.subgraph(comps[0]).copy()
LCC_r = G_r.subgraph(comps_r[0]).copy()
LCC_nr = G_nr.subgraph(comps_nr[0]).copy()
density(LCC), density(LCC_r), density(LCC_nr)
# Centrality Measures
LCC_ary = [LCC_r, LCC_nr]
DC, EVC = [], []
for g in LCC_ary:
  DC.append(nx.degree_centrality(g))
  EVC.append(nx.eigenvector_centrality(g, max_iter=10000))

df_d, df_e = pd.DataFrame(DC), pd.DataFrame(EVC)
# Row descriptive stats
df_e.apply(pd.DataFrame.describe, axis=1)

degree = df_d.mean(axis=0).sort_values(ascending=False)
between = df_b.mean(axis=0).sort_values(ascending=False)
eigen = df_e.mean(axis=0).sort_values(ascending=False)
close = df_c.mean(axis=0).sort_values(ascending=False)
centrality = pd.concat([degree, eigen, between, close], axis=1)
centrality.rename(columns={0:"Degree", 1:"Eigenvector", 2:"Betweenness", 3:"Closeness").head(20)

### Plot of clustering coefficients
cc = nx.clustering(G)
cc_r = nx.clustering(G_r)
cc_nr = nx.clustering(G_nr)
data = np.array(list(cc.values()))
data_r = np.array(list(cc_r.values()))
data_nr = np.array(list(cc_nr.values()))
# x = data_r 
# y = data_nr
# bins = np.linspace(0, 1, 20)
# plt.hist([x, y], bins, alpha=0.75, label=['G_nr', 'G_r'])
# plt.legend(loc='upper right')
# plt.xlabel('Clustering coefficient')
# plt.ylabel('Frequency count')
# plt.show()

# Ratio of 0 to 1 clustering coefficients
x = [data, data_r, data_nr]
for d in x:
  np.count_nonzero(d == 0)/len(d)
  np.count_nonzero(d == 1)/len(d)


