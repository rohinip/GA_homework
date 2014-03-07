import pca_tutorial as pca

data_raw, reduced_dim_data, reconstructed_data = pca.test_plot("./pca_test_set.txt", top_n_features=1 )

print "** show 5 **"
print "data_raw"
print data_raw[:5,:]
print "reduced_dim_data"
print reduced_dim_data[:5,:]
print "reconstructed_data"
print reconstructed_data

print reconstructed_data[:5,:]
