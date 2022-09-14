from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

digits.data

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = clusters == i
    labels[mask] = mode(digits.target[mask])[0]

np.zeros_like(clusters)

mask = clusters == 1
print("which images are clustered into cluster 1?") 
print(mask)
print("")
print("What is the length of mask? ")
print(len(mask))
print("")
print("How many images are clustered into cluster 1?")
print(mask.sum())

print("Among the images that are clustered into cluster 1, what is the most possible digit?")
print("")
print(mode(digits.target[mask])[0])

from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)

from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()  # for plot styling
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, 
            fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');



