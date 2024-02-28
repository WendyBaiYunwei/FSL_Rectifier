import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the embeddings and labels from pickle files
def load_data_from_pickle(embeddings_file, labels_file):
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)
    return embeddings, labels

# Perform t-SNE on the embeddings
def perform_tsne(embeddings, perplexity=30, n_iter=4000, random_state=1): 
    tsne = TSNE(n_iter=n_iter, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

if __name__ == "__main__":
    embeddings_pickle_path = "embeddings_overall.pkl"
    labels_pickle_path = "embeddings_label_overall.pkl"

    embeddings, labels = load_data_from_pickle(embeddings_pickle_path, labels_pickle_path)
    embeddings = embeddings.reshape(-1, 64)
    postns = perform_tsne(embeddings, random_state=0) # len, 2
        
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

    plt.rcParams['font.family'] = 'Times New Roman'
    # keep_idx = x_values[x_values < max_x] and y_values[y_values < max_y] and\
    #     x_values[x_values > min_x] and y_values[y_values < max_y]
    cur_postns = postns
        
    for i in range(num_classes):
        indices = (labels == unique_labels[i]).reshape(-1)
        plt.scatter(cur_postns[indices, 0], cur_postns[indices, 1], c=[colors[i]], s=4, alpha=0.3)

    plt.xticks([]) 
    plt.yticks([])
    
    plt.savefig('tsne_overall_after.png')
    # plt.savefig('tsne.pdf')
