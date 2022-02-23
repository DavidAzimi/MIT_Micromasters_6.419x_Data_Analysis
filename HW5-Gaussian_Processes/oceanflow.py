from numpy import nan
import matplotlib.pyplot as plt
from numpy import linspace
from numpy import isnan
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns;
sns.set_theme ()
from sklearn.neighbors import KernelDensity

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
M = np.sqrt(h**2+v**2) # Compute speed flow
# var = np.var(M, axis=0)
# var = var.astype(float)
# var[var == 0] = np.nan # Remove zero variance
# np.unravel_index(np.nanargmin(var, axis=None), var.shape) # Min speed flow
# np.unravel_index(np.argmax(h, axis=None), h.shape) # Max horizontal flow
# np.mean(h), np.mean(v)

# # Helper function returns velocity based on input location and time
# def velocity(location, time):
#     if location # out of bounds:
#         location = #argmin of euclidean distance serrogate loc
#     v = # function of loc and t
#     return v

# Helper function returns euclidean distance between two points
def distance(points):
    point1, point2 = points[0], points[1]
    y1, x1 = point1[0], point1[1]
    y2, x2 = point2[0], point2[1]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# Helper function to round location to integer point and snap it to the grid if it goes out of bounds
def clip(location):
    _, y_lim, x_lim = h.shape
    if location[0] < 0: # negative case
        location[0] = 0 
    elif location[0] > y_lim-1: # positive case
        location[0] = y_lim-1
    if location[1] < 0: # negative case
        location[1] = 0 
    elif location[1] > x_lim-1: # positive case
        location[1] = x_lim-1
    return location

##################
### SIMULATION ###
##################
# Velocity changes depending on the location of the particle
T =  132 # total time
TIME = T//3
N =  100 # number of iterations
# fix number of iterations, epsilon = T / N
GRID_SIZE = 3. # length
EPSILON = 3. 
# Fix the time increments, N = T / epsilon
NUM_POINTS = 10**3
# Map of land/water
mask_file = 'OceanFlow/mask.csv'
mask_data = np.loadtxt(mask_file, delimiter=',')
# # Random initialization
# range_x = (0, h.shape[2]-1)
# range_y = (0, h.shape[1]-1)
# Plane Crash Initialization
mu = (350, 100)
cov1 = [[100,0], [0,100]]
cov2 = [[10,0], [0,10]]
cov3 = [[1,0], [0,1]]
cov = cov1
y, x = np.random.multivariate_normal(mu, cov, 1000).T
# Start Simulation
rand_points = []
land_points = []
for i in range(NUM_POINTS):
#     x = np.random.randint(*range_x)
#     y = np.random.randint(*range_y)
    ver = y[i]
    hor = x[i]
    if mask_data[int(np.round(ver)),int(np.round(hor))]: 
        rand_points.append(np.array([float(ver),float(hor)]))
    else:
        land_points.append(np.array([float(ver),float(hor)]))


# Color Map
colors = [ cm.viridis(x) for x in linspace(0, 1, len(rand_points)) ]
#fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(mask_data.T, origin='lower', cmap='binary_r', interpolation='nearest')
for t in range(TIME):
    speed_flow = []
    for i in range(len(rand_points)):
        loc = tuple(map(int, np.round(rand_points[i]))) 
        speed_flow.append(M[t][loc])
        rand_points[i] += np.array([h[t][loc], v[t][loc]])# * epsilon/grid_size is 1 in this case
        rand_points[i] = clip(rand_points[i])
# Plotting simulation
    plt.scatter(*zip(*rand_points), s=np.multiply(50,speed_flow), color=colors, alpha=.99*t**2/TIME**2+.01)
# 
# ## Uncomment lines to show land points in red
# #    speed_flow = []
# #    for i in range(len(land_points)):
# #        loc = tuple(map(int, np.round(land_points[i]))) 
# #        speed_flow.append(M[t][loc])
# #        land_points[i] += np.array([h[t][loc], v[t][loc]])# * epsilon/grid_size is 1 in this case
# #        land_points[i] = clip(land_points[i])
# #    plt.scatter(*zip(*land_points), s=np.multiply(50,speed_flow), color='red', alpha=.99*t**2/TIME**2+.01)
    txt =plt.gcf().text(0.5, 0.02, f'Time = {t*3} hours', fontsize=14)
    plt.pause(0.0001)
    txt.remove()

plt.show()

### KERNEL DENSITY PLOTS
# From https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
# fig, ax = plt.subplots(1,3)
# time_stamps = ['48 hours', '72 hours', '120 hours']
# cmap = 'Blues'
# saved_points = np.array(saved_points)
# for i, axi in enumerate(ax):
#     axi.set_title(time_stamps[i])
#     axi.imshow(mask_data.T, origin='lower', cmap='binary_r', interpolation='nearest')
#     sns.kdeplot(x=np.array(saved_points[i])[:,0], y=np.array(saved_points[i])[:,1], shade=True, thresh=0.05, ax=axi)
# 
# plt.show()
 #    # construct a spherical kernel density estimate of the distribution
 #    kde = KernelDensity(bandwidth=0.03, metric='haversine')
 #    kde.fit(saved_points[i]))

 #    # evaluate only in the ocean
 #    Z = np.full(mask_data.shape[0], -9999.0)
 #    Z[land_mask] = np.exp(kde.score_samples(xy))
 #    Z = Z.reshape(X.shape)

 #    # plot contours of the density
 #    levels = np.linspace(0, Z.max(), 25)
 #    axi.contourf(X, Y, Z, levels=levels, cmap=cmaps[i])
    
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(mask_data.T, origin='lower', cmap='binary_r', interpolation='nearest')
# for t in range(TIME):
#     speed_flow = []
#     for i in range(len(rand_points)):
#         loc = tuple(map(int, np.round(rand_points[i]))) 
#         speed_flow.append(M[t][loc])
#         rand_points[i] += np.array([h[t][loc], v[t][loc]])# * epsilon/grid_size is 1 in this case
#         rand_points[i] = clip(rand_points[i])
#     plt.scatter(*zip(*rand_points), s=np.multiply(50,speed_flow), color=colors, alpha=.99*t**2/TIME**2+.01)

# plt.show()

