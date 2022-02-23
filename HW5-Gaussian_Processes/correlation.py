from numpy import nan
import matplotlib.pyplot as plt
from numpy import linspace
from numpy import isnan
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns;
sns.set_theme ()

### Import Data
vertical, horizontal = [], []
for i in range(1,101):
    print(i)
    horizontal.append(pd.read_csv('./OceanFlow/' + str(i) + 'u.csv', header=None))
    vertical.append(pd.read_csv('./OceanFlow/' + str(i) + 'v.csv', header=None))

h = np.stack(horizontal)
v = np.stack(vertical)

# ### Explore Data
# h.shape, v.shape # ((100, 504, 555), (100, 504, 555))
# M = np.sqrt(h**2+v**2) # Compute speed flow
# var = np.var(M, axis=0)
# var = var.astype(float)
# var[var == 0] = np.nan # Remove zero variance
# np.unravel_index(np.nanargmin(var, axis=None), var.shape) # Min speed flow
# np.unravel_index(np.argmax(h, axis=None), h.shape) # Max horizontal flow
# np.mean(h), np.mean(v)

# Helper function returns euclidean distance between two points
def distance(points):
    point1, point2 = points[0], points[1]
    y1, x1 = point1[0], point1[1]
    y2, x2 = point2[0], point2[1]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

###################
### CORRELATION ###
##################
# My naive approach to correlation
NUM_PAIRS = 10**6
# Make zero variance values nan
var=np.var(h, axis=0)
var = var.astype(float)
for i in range(h.shape[0]):
    h[i][var == 0] = np.nan
# Create random point pairs
index_y1 = np.random.choice(h[0].shape[0], NUM_PAIRS, replace=True)
index_x1 = np.random.choice(h[0].shape[1], NUM_PAIRS, replace=True)
index_y2 = np.random.choice(h[0].shape[0], NUM_PAIRS, replace=True)
index_x2 = np.random.choice(h[0].shape[1], NUM_PAIRS, replace=True)
# Create dict with point pair key ((y1,x1),(y2,x2)) and correlation value
correlations = {} 
for i in range(NUM_PAIRS):
    y_1, x_1, y_2, x_2 = index_y1[i], index_x1[i], index_y2[i], index_x2[i]
    correlations[((y_1, x_1), (y_2, x_2))] = np.corrcoef(h[:, y_1, x_1], h[:, y_2, x_2])
# print(correlations)    # be patient this is NUM_PAIRS
# Remove nan values
# clean_dict = filter(lambda k: not isnan(correlations[k][0,1]), correlations)
clean_dict = {k: correlations[k] for k in correlations if not isnan(correlations[k][0,1])}
clean_dict = {k: clean_dict[k] for k in clean_dict if not clean_dict[k][0,1]==1}
sort_corr = sorted(clean_dict.items(), key=lambda x: x[1][0,1], reverse=True)

####################################
### Plot long-range correlations ###
####################################
# Map of land/water
mask_file = 'OceanFlow/mask.csv'
mask_data = np.loadtxt(mask_file, delimiter=',')
threshold_pos = 0.96
threshold_neg = -0.96

lr_pos = []
lr_neg = []
for cor in sort_corr:
    if distance(cor[0]) > 50:
        if cor[1][0,1] > threshold_pos:
            lr_pos.append([cor[0], cor[1][0,1]])
        elif cor[1][0,1] < threshold_neg:
            lr_neg.append([cor[0], cor[1][0,1]])


fig, (ax1,ax2) = plt.subplots(1,2)
# Positive horizontal correlations
#ax1 = sns.heatmap(mask_data, cbar=False)
ax1.imshow(mask_data, cmap='binary_r', interpolation='nearest')
colors_pos = [ cm.Reds(x) for x in linspace(0.1, 1, len(lr_pos)) ]
for i in range(0, len(lr_pos)):
    ax1.plot([lr_pos[i][0][0][1], lr_pos[i][0][1][1]], [lr_pos[i][0][0][0], lr_pos[i][0][1][0]], 'ro-', color=colors_pos[i])

ax1.set_title(f"Long range positive (r > {threshold_pos}) horizontal correlations)")
ax1.grid(False)
# Negative horizontal correlations
ax2.imshow(mask_data, cmap='binary_r', interpolation='nearest')
colors_neg = [ cm.Blues(x) for x in linspace(0.1, 1, len(lr_neg)) ]
for i in range(0, len(lr_neg)):
    ax2.plot([lr_neg[i][0][0][1], lr_neg[i][0][1][1]], [lr_neg[i][0][0][0], lr_neg[i][0][1][0]], 'ro-', color=colors_neg[i])

ax2.set_title(f"Long range negative (r < {threshold_neg}) horizontal correlations)")
ax2.grid(False)
plt.show()

# Ref for correlation matrix
# https://www.geeksforgeeks.org/sort-correlation-matrix-in-python/
my_array = h[:, index_y1, index_x1]
df = pd.DataFrame(my_array)

# Create correlation matrix
corr_mat = df.corr(method='pearson')
  
# Convert correlation matrix to 1-D Series and sort
sorted_mat = corr_mat.unstack().sort_values()
  
print(sorted_mat)

# Retain upper triangular values of correlation matrix and
# make Lower triangular values Null
upper_corr_mat = corr_mat.where(
    np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
  
# Convert to 1-D series and drop Null values
unique_corr_pairs = upper_corr_mat.unstack().dropna()
  
# Sort correlation pairs
sorted_mat = unique_corr_pairs.sort_values()

print(sorted_mat)
