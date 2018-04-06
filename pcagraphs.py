import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

mean = [3, 4]
cov = [[1, 5], [5,  10]]
eigvals, eigvecs = np.linalg.eig(cov)
print(eigvals)
print(eigvecs)

X = np.random.multivariate_normal(mean, cov, 100).T

fig = plt.figure()

ax1 = fig.add_subplot(131)
ax1.scatter(X[0], X[1])
ax1.grid()
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.axvline(0, c='k')
ax1.axhline(0, c='k')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_title('Original data')

mean = np.mean(X)
print('mean', mean)
centred = X - mean

ax2 = fig.add_subplot(132)
ax2.scatter(centred[0], centred[1])
ax2.grid()
ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)
ax2.axvline(0, c='k')
ax2.axhline(0, c='k')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.set_title('Step 1: Centre data around mean')


ranges = np.ptp(X)
print('ranges', ranges)
scaled = np.divide(centred, ranges)

ax3 = fig.add_subplot(133)
ax3.scatter(scaled[0], scaled[1])
ax3.grid()
ax3.set_xlim(-1, 1)
ax3.set_ylim(-1, 1)
ax3.axvline(0, c='k')
ax3.axhline(0, c='k')
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.set_title('Step 2: Scale data')

plt.show()

covscaled = np.matmul(scaled, scaled.T)
print('\n\nScaled feature covariance')
print(covscaled)
eigvals, eigvecs = np.linalg.eig(covscaled)
print('eigvals')
print(eigvals)
print('Eigvecs')
print(eigvecs)

fig2 = plt.figure()
ax4 = fig2.add_subplot(131)
ax4.grid()
ax4.scatter(scaled[0], scaled[1])
ax4.set_xlim(-1, 1)
ax4.set_ylim(-1, 1)
ax4.quiver(eigvals[0]*eigvecs[0,0], eigvals[0]*eigvecs[1, 0], scale_units='xy', scale=5)
ax4.quiver(eigvals[1]*eigvecs[0,1], eigvals[1]*eigvecs[1, 1], scale_units='xy', scale=5)
ax4.axvline(0, c='k')
ax4.axhline(0, c='k')
ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
ax4.set_title('Step 3: Determine eigenvectors of covariance matrix')

eigvecs = eigvecs[:, -1::-1]
transformed = np.matmul(eigvecs.T, scaled)

ax5 = fig2.add_subplot(132)
ax5.grid()
ax5.scatter(transformed[0], transformed[1])
ax5.set_xlim(-1, 1)
ax5.set_ylim(-1, 1)
ax5.axvline(0, c='k')
ax5.axhline(0, c='k')
ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
ax5.yaxis.set_major_locator(MaxNLocator(integer=True))
ax5.set_title('Step 4: Make eigenvectors the axes')

reduced = np.matmul(eigvecs[:, 0], scaled)

ax6 = fig2.add_subplot(133)
ax6.grid()
ax6.scatter(reduced, np.zeros_like(reduced))
ax6.set_xlim(-1, 1)
ax6.set_ylim(-1, 1)
ax6.axvline(0, c='k')
ax6.axhline(0, c='k')
ax6.xaxis.set_major_locator(MaxNLocator(integer=True))
ax6.yaxis.set_major_locator(MaxNLocator(integer=True))
ax6.set_title('Step 5: Throw away higher principal components')
plt.show()