import matplotlib
matplotlib.use("AGG")
from matplotlib import pyplot as plt
import numpy as np

def pca(data_raw, top_n_features=9999999):
    # for each feature vector in the matrix, determine the mean,
    data_mean = np.mean(data_raw, axis=0)
    # then remove (subtract) it from the vector
    data_mean_removed = data_raw - data_mean

    # determine the covariance matrix for the "normalized" matrix
    covariance_matrix = np.cov(data_mean_removed, rowvar=0)

    # obtain the eigenvalues and eigenvectors for the covariance matrix
    #NOTE: convert to matrix to make (matrix) math below easier
    eigen_vals,eigen_vects = np.linalg.eig(np.mat(covariance_matrix))

    # generate an sorted array of the indices of the eigenvalue array as sorted by value
    eigen_value_indices = np.argsort(eigen_vals)
    # truncate this index by the number of "top" features desired
    eigen_value_indices_top_n = eigen_value_indices[:-(top_n_features+1):-1]

    # obtain the eigenvectors for the "top" features by eigenvalue ( the reduced set of eignenvectors )
    reduced_eigen_vects = eigen_vects[:,eigen_value_indices_top_n]

    # now transform data into new dimensions
    reduced_dim_data = data_mean_removed * reduced_eigen_vects

    reconstructed_data = (reduced_dim_data * reduced_eigen_vects.T) + data_mean

    return (reduced_dim_data, reconstructed_data)



# function to load delimited data
def load_data_file(fileName, delim=','):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr if not line[0].startswith("#") ]
    return np.mat(datArr)


# wrapper to load data and make pca call
def test(filename="", top_n_features=999999) :
    data_raw = load_data_file(filename)

    # reduced_dim_data contains the matrix in our reduced dimensions, which should be "top_n_features" dimensions
    reduced_dim_data, reconstructed_data = pca(data_raw, top_n_features)

    return (data_raw, reduced_dim_data, reconstructed_data)


# wrapper for pca call with plotting (NOTE only for 2-d data)
def test_plot(filename="", top_n_features=999999) :

    import matplotlib.pyplot as plt

    data_raw, reduced_dim_data, reconstructed_data = test(filename, top_n_features)

    fig = plt.figure(0)

    ax = fig.add_subplot(121)
    # plot the original data
    # NOTE: A to extract the base array
    ax.scatter(data_raw[:,0].flatten().A[0], data_raw[:,1].flatten().A[0], marker='^', s=40)

    # plot the reconstructed data relative our existing coordinate system
    ax.scatter(reconstructed_data[:,0].flatten().A[0], reconstructed_data[:,1].flatten().A[0], marker='o', s=30, c='red')

    # separately plot the new coordinate system data
    ax2 = fig.add_subplot(122)
    ax2.scatter(reduced_dim_data[:,0].flatten().A[0], np.zeros(reduced_dim_data[:,0].flatten().A[0].shape[0]), marker='o', s=30, c='green')

    plt.savefig("pca_tutorial.png")

    return (data_raw, reduced_dim_data, reconstructed_data)
