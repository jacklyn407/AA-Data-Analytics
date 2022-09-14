from sklearn.decomposition import PCA

pca = PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

accum_explained_var = np.cumsum(pca.explained_variance_ratio_)

min_threshold = np.argmax(accum_explained_var > 0.90) # use 90%

min_threshold

pca = PCA(n_components = min_threshold + 1)

X_train_projected= pca.fit_transform(X_train)
X_test_projected = pca.transform(X_test)

X_train_projected.shape

