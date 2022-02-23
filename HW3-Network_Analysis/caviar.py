import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

### Import files as pd dataframes and networkx graphs
phases = {}
G = {}
player_names = pd.read_csv('./res/CAVIAR_names.csv')
players = ['n1', 'n3', 'n5', 'n6', 'n8', 'n11', 'n12', 'n16', 'n17', 'n33', 'n76', 'n77', 'n80', 'n82', 'n83', 'n84', 'n85', 'n86', 'n87', 'n88', 'n89', 'n96', 'n106']
key_players = {'n1', 'n12', 'n3', 'n76', 'n87', 'n41'}
for i in range(1,12): 
  var_name = "phase" + str(i)
  file_name = r"./CAVIAR/" + var_name + ".csv"
  phases[i] = pd.read_csv(file_name, index_col=0) #index_col = ["players"]
  phases[i].columns = "n" + phases[i].columns
  phases[i].index = phases[i].columns
  # undirected
  #G[i] = nx.from_pandas_adjacency(phases[i])
  # directed
  G[i] = nx.from_pandas_adjacency(phases[i], create_using=nx.DiGraph)
  G[i].name = var_name

### Draw one network
# g = G[2]
# nx.draw(g, pos=nx.drawing.nx_agraph.graphviz_layout(g), with_labels=True) 
# plt.show()

### Draw all network phases
# fig, axes = plt.subplots(nrows=4, ncols=3)
# ax = axes.flatten()
# for i in range(1,12): 
#   g = G[i]
#   colors = []
#   for node in g:
#     if node in key_players:
#       colors.append("yellow")
#     else: colors.append("#3399ff")
#   nx.draw(g, pos=nx.drawing.nx_agraph.graphviz_layout(g), node_color=colors, ax=ax[i-1], with_labels=True)
#   ax[i-1].set_axis_off()
#   ax[i-1].set_title(f'Phase {i}')
# ax[11].set_axis_off()
# plt.show()

### HITS (Hubs and Athorities Directed Graphs)
Hubs, Auth = [], []
for i in range(1,12):
  largest_cc = max(nx.weakly_connected_components(G[i]), key=len)
  g = G[i].subgraph(largest_cc)
  Hubs.append(nx.hits(g, max_iter=1000000)[0])
  Auth.append(nx.hits(g, max_iter=1000000)[1])
#  for g in [G[i].subgraph(max(c) for c in nx.weakly_connected_components(G[i])]:
#  by component
#    Hubs.append(nx.hits(g)[0]) 
#    Auth.append(nx.hits(g)[1])
#    print(g)
df_H, df_A = pd.DataFrame(Hubs), pd.DataFrame(Auth)
key_players=['n1', 'n3', 'n12', 'n76', 'n83','n89', 'n82', 'n87', 'n41']
print(df_H[key_players].round(5))
print(df_A[key_players].round(5))

### Centrality
#for i in range(1,12):
  #print(i, nx.degree_centrality(G[i]))
  #print(i, nx.betweenness_centrality(G[i], normalized = True))
  #print(i, nx.eigenvector_centrality(G[i]))

### Create names dict
# player_dict = {}
# player_names["desc"] = player_names["full_name"] + ', ' + player_names["description"]
# for idx in range(len(player_names["player"])):
#   player_dict[player_names["player"][idx]] = player_names["desc"][idx]

### Reformat names df
# idx = player_names["player"]
# player_names = player_names.rename(index=index)
# player_names.drop(['player', 'desc'], axis=1)

### Temporal Consistency
# DC, BWC, EVC = [], [], []
# for i in range(1,12):
#   DC.append(nx.degree_centrality(G[i]))
#   BWC.append(nx.betweenness_centrality(G[i], normalized = True))
#   EVC.append(nx.eigenvector_centrality(G[i]))
# df_d, df_b, df_e = pd.DataFrame(DC), pd.DataFrame(BWC), pd.DataFrame(EVC)
# # Named players
# degree_named_NA = df_d[players].mean(axis=0).sort_values(ascending=False)
# between_named_NA = df_b[players].mean(axis=0).sort_values(ascending=False)
# eigen_named_NA = df_e[players].mean(axis=0).sort_values(ascending=False)
# centrality_named_NA = pd.concat([between_named_NA, eigen_named_NA, degree_named_NA, player_names], axis=1)
# centrality_named_NA.rename(columns={1:"Eigenvector", 0:"Betweenness", 2:"Degree", "full_name":"Name", "description":"Role"})
# # All players
# degree_all_NA = df_d.mean(axis=0).sort_values(ascending=False)
# between_all_NA = df_b.mean(axis=0).sort_values(ascending=False)
# eigen_all_NA = df_e.mean(axis=0).sort_values(ascending=False)
# centrality_all_NA = pd.concat([between_all_NA, eigen_all_NA, degree_all_NA, player_names], axis=1)
# centrality_all_NA.rename(columns={1:"Eigenvector", 0:"Betweenness", 2:"Degree", "full_name":"Name", "description":"Role"}).head(20)
# # Impute missing values with 0
# df_d0 = df_d.fillna(0)
# df_b0 = df_b.fillna(0)
# df_e0 = df_e.fillna(0)
# # Named players
# degree_named_0 = df_d0[players].mean(axis=0).sort_values(ascending=False)
# between_named_0 = df_b0[players].mean(axis=0).sort_values(ascending=False)
# eigen_named_0 = df_e0[players].mean(axis=0).sort_values(ascending=False)
# centrality_named_0 = pd.concat([between_named_0, eigen_named_0, degree_named_0, player_names], axis=1)
# centrality_named_0.rename(columns={1:"Eigenvector", 0:"Betweenness", 2:"Degree", "full_name":"Name", "description":"Role"})
# # All players
# degree_all_0 = df_d0.mean(axis=0).sort_values(ascending=False)
# between_all_0 = df_b0.mean(axis=0).sort_values(ascending=False)
# eigen_all_0 = df_e0.mean(axis=0).sort_values(ascending=False)
# centrality_all_0 = pd.concat([between_all_0, eigen_all_0, degree_all_0, player_names], axis=1)
# centrality_all_0.rename(columns={1:"Eigenvector", 0:"Betweenness", 2:"Degree", "full_name":"Name", "description":"Role"}).head(20)
# Impute missing values from phase 1 and 2 with mean from 3-5
#df_d.iloc[:,0:2].fillna(df_d.iloc[:,3:5].isna().mean(axis=0), inplace=True)


### Plot number of nodes and edges
# x=[]
# n=[]
# e=[]
# for i in range(1,12):
#   x.append(i)
#   n.append(nx.number_of_nodes(G[i]))
#   e.append(nx.number_of_edges(G[i]))
# plt.plot(x,n,label="Nodes")
# plt.plot(x,e,label="Edges")
# plt.xticks(x)
# plt.xlabel('Phase Number')
# plt.ylabel('Frequency')
# plt.title('Nodes and Edges of the CAVIAR criminal network over 11 phases (1994-1996)')
# plt.legend()
# plt.grid(linewidth=1, alpha=.4)
# plt.show()
# print(x,'\n',n,'\n',e)
# #print(nx.number_of_edges(G[1]))
