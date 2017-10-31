from numpy import genfromtxt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

fn = r'C:\Users\DELL I5558\Desktop\Python\NSW-ER01.csv'
my_data = genfromtxt(fn, delimiter=',')

scaler = StandardScaler()
model = KMeans(n_clusters=4)
pipeline = make_pipeline(scaler, model)
scaler.fit(my_data)
my_data_scaled = scaler.transform(my_data)
model.fit(my_data_scaled)
labels = model.predict(my_data_scaled)

model = TSNE(n_components=2, perplexity=40, learning_rate=100, n_iter=1000, random_state=0)
transformed = model.fit_transform(my_data_scaled)
xs = transformed[:, 0]
ys = transformed[:, 1]

plt.scatter(xs, ys, c=labels)
plt.show()
