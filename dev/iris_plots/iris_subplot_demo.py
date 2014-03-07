import matplotlib
matplotlib.use("AGG")
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Load the data with load_iris from sklearn
data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']

# Instantiate the top-level figure for this plotting exercise
fig = plt.figure(0)

# Commented out code immediately below is alternate way of
#   creating subplots. Ignore unless you want to explore
#   on your own.
#
#axes1 = plt.subplot2grid((2, 3), (0, 0))
#axes2 = plt.subplot2grid((2, 3), (0, 1))
#axes3 = plt.subplot2grid((2, 3), (0, 2))
#axes4 = plt.subplot2grid((2, 3), (1, 0))
#axes5 = plt.subplot2grid((2, 3), (1, 1))
#axes6 = plt.subplot2grid((2, 3), (1, 2))

# Instantiate our six subplots. Note that context will be set to the last one by default.
ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2)
ax3 = fig.add_subplot(2,3,3)
ax4 = fig.add_subplot(2,3,4)
ax5 = fig.add_subplot(2,3,5)
ax6 = fig.add_subplot(2,3,6)

# Title our figure. This is a different command than titling an individual subplot.
fig.suptitle('Subplots demo with Iris Data', fontsize=14)

# Sets the number of decimal places in our y-axis tick labels
majorFormatter = FormatStrFormatter('%1.1f')

# Loop through our subplots and set font sizes
# HINT: plt.gcf().axes gets all the axes for our current figure
all_axes = plt.gcf().axes
for ax in all_axes:
    ax.yaxis.set_major_formatter(majorFormatter)
    for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
        ticklabel.set_fontsize(6)

# Plot and format first subplot. We could make this a loop, but doing it manually for each
#  is more readable for learning purposes.
for t,marker,c in zip(xrange(3),">ox","rgb") :
    # We plot each class on its own to get different colored markers
    ax1.scatter(features[target == t,0],
                features[target == t,1],
                marker=marker,
                c=c)
    ax1.set_xlabel(feature_names[0], fontsize=8)
    ax1.set_ylabel(feature_names[1], fontsize=8)

# Plot and format second subplot. We could make this a loop, but doing it manually for each
#  is more readable for learning purposes.

for t,marker,c in zip(xrange(3),">ox","rgb") :
    ax2.scatter(features[target == t,0],
                features[target == t,2],
                marker=marker,
                c=c)
    ax2.set_xlabel(feature_names[0], fontsize=8)
    ax2.set_ylabel(feature_names[2], fontsize=8)

# Plot and format third subplot. We could make this a loop, but doing it manually for each
#  is more readable for learning purposes.

for t,marker,c in zip(xrange(3),">ox","rgb") :
    ax3.scatter(features[target == t,0],
                features[target == t,3],
                marker=marker,
                c=c)
    ax3.set_xlabel(feature_names[0], fontsize=8)
    ax3.set_ylabel(feature_names[3], fontsize=8)

# Plot and format fourth subplot. We could make this a loop, but doing it manually for each
#  is more readable for learning purposes.

for t,marker,c in zip(xrange(3),">ox","rgb") :
    ax4.scatter(features[target == t,1],
                features[target == t,2],
                marker=marker,
                c=c)
    ax4.set_xlabel(feature_names[1], fontsize=8)
    ax4.set_ylabel(feature_names[2], fontsize=8)

# Plot and format fifth subplot. We could make this a loop, but doing it manually for each
#  is more readable for learning purposes.

for t,marker,c in zip(xrange(3),">ox","rgb") :
    ax5.scatter(features[target == t,1],
                features[target == t,3],
                marker=marker,
                c=c)
    ax5.set_xlabel(feature_names[1], fontsize=8)
    ax5.set_ylabel(feature_names[3], fontsize=8)

# Plot and format sixth subplot. We could make this a loop, but doing it manually for each
#  is more readable for learning purposes.

for t,marker,c in zip(xrange(3),">ox","rgb") :
    ax6.scatter(features[target == t,2],
                features[target == t,3],
                marker=marker,
                c=c)
    ax6.set_xlabel(feature_names[2], fontsize=8)
    ax6.set_ylabel(feature_names[3], fontsize=8)

# Save our figure with subplots as one file
plt.savefig('iris_plot_combined.png')
