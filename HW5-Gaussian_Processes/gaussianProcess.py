import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import seaborn as sns
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
M = np.sqrt(h**2+v**2) # Compute speed flow

# Choose point
point = (300, 400)
point2 = (150, 150)
point3 = (150, 400)
point4 = (267, 400)
velocity_x = h[:, point[0], point[1]].reshape(-1,1) 
velocity_y = v[:, point[0], point[1]].reshape(-1,1)
# sns.histplot(velocity_x)
# plt.xlabel('Velocity_x')
# sns.histplot(velocity_y)
# plt.xlabel('Velocity_y')

# # RBF Kernel
# def rbf_kernel(z_i, z_j, sigma, l):
#     return sigma**2 * np.exp(-(z_i - z_j)**2 / l**2)

# Sampling from GP
# https://katbailey.github.io/post/gaussian-processes-for-dummies/
# Define the kernel function
def kernel(a, b, sigma, l):
#    sqdist = np.sum(a**2) + np.sum(b**2) - 2*np.dot(a, b.T)
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return sigma * np.exp(-.5 * (1/l**2) * sqdist)

#sigma = 0.2
#l = 2
#K_ss = rbf_kernel(velocity_x, velocity_x, .2, 2)
# Kx_ss = kernel(velocity_x, velocity_x, sigma, l)
# Ky_ss = kernel(velocity_y, velocity_y, sigma, l)
# 
# # Get cholesky decomposition (square root) of the
# # covariance matrix
# Lx = np.linalg.cholesky(Kx_ss + 1e-15*np.eye(len(Kx_ss)))
# Ly = np.linalg.cholesky(Ky_ss + 1e-15*np.eye(len(Ky_ss)))
# # Sample 3 sets of standard normals for our test points,
# # multiply them by the square root of the covariance matrix
# fx_prior = np.dot(Lx, np.random.normal(size=(len(Kx_ss),3)))
# fy_prior = np.dot(Ly, np.random.normal(size=(len(Ky_ss),3)))
# 
# # Now let's plot the 3 sampled functions.
# plt.plot(velocity_x, fx_prior)
# #pl.axis([-5, 5, -3, 3])
# plt.title('Three samples from the GP prior')
# plt.xlabel('velocity_x')
# plt.show()
# plt.plot(velocity_y, fy_prior)
# #pl.axis([-5, 5, -3, 3])
# plt.title('Three samples from the GP prior')
# plt.xlabel('velocity_y')
# plt.show()

kf = KFold(n_splits=10)
n = 10
lengths = np.linspace(0.1, 5, n).reshape(-1,1)
sigmas = np.linspace(.01, .5, n).reshape(-1,1)
velocities = [velocity_x, velocity_y]

for l in lengths:
    for sigma in sigmas:
        for train_idx, test_idx in kf.split(velocity_x):
            for velocity in velocities:
                train, test = velocity[train_idx], velocity[test_idx]
                K_ss = kernel(test, test, sigma, l)
                # Apply the kernel function to our training points
                K = kernel(train, train, sigma, l)
                L = np.linalg.cholesky(K + 0.00005*np.eye(len(train)))
                # Compute the mean at our test points.
                K_s = kernel(train, test, sigma, l)
                Lk = np.linalg.solve(L, K_s)
                mu = np.dot(Lk.T, np.linalg.solve(L, train)).reshape((n,))
                # Compute the standard deviation so we can plot it
                s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
                stdv = np.sqrt(s2)
                # Draw samples from the posterior at our test points.
                L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
                f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))

                plt.plot(train, train, 'bs', ms=8)
                plt.plot(test, f_post)
                plt.gca().fill_between(test.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
                plt.plot(test, mu, 'r--', lw=2)
                plt.title('Three samples from the GP posterior')
                plt.show()

