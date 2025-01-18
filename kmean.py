import os
import numpy as np
import faiss
from tqdm import tqdm
def load_embeddings(npz_path):
    dataset = find_all_npz(npz_path=npz_path)
    # dataset = [os.path.join(npz_path, npz) for npz in os.listdir(npz_path) if npz.endswith('.npz')]
    embeddings = []
    keys = []
    for npz in tqdm(dataset):
        data = np.load(npz)
        if len(data['embeddings'].shape) == 2:
            embeddings.append(data['embeddings'])
            keys.append(data['ids'])
        # else:
            # print(data['embeddings'].shape)
    # print(embeddings[1].shape)
    embeddings = np.vstack(embeddings)
    keys = np.concatenate(keys)
    return embeddings, keys

def perform_kmeans(embeddings, n_clusters, n_iter=100, gpu=False):
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, verbose=True, gpu=gpu)
    kmeans.train(embeddings)
    cluster_centers = kmeans.centroids
    distance, labels = kmeans.index.search(embeddings, 1)

    distance = distance.flatten()
    labels = labels.flatten()
    return cluster_centers, labels, distance

def sample_from_clusters(embeddings, keys, labels, distance, n_samples=5):
    sampled_keys = []
    sampled_labels = []
    unique_labels = np.unique(labels)
    label_counts = {}

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        label_counts[label] = len(indices)
        sorted_indices = indices[np.argsort(distance[indices])]

        if len(sorted_indices) > n_samples:
            # sampled_indices = sorted_indices[:n_samples] #最近采样
            sampled_indices = np.random.choice(indices, n_samples, replace=False) #random 采样
        else:
            sampled_indices = sorted_indices

        sampled_keys.append(keys[sampled_indices])
        sampled_labels.extend([label] * len(sampled_indices))
    # print(len(sampled_labels))
    sampled_keys = np.concatenate(sampled_keys)
    sampled_labels = np.array(sampled_labels)

    return sampled_keys, sampled_labels, label_counts

def read_from_npz():
    npz_file = '/workspace/Crilias/zhangzhenxing/data/clustering/sampled_results.npz'

    data = np.load(npz_file)

    keys = []
    labels = []
    label_counts = []

    keys.append(data['sampled_keys'])
    labels.append(data['sampled_labels'])

    keys = np.concatenate(keys)
    labels = np.array(labels)

    label_keys = data['label_keys']
    label_values = data['label_values']
    label_counts = dict(zip(label_keys, label_values))

    print("Label counts:", label_counts[0])#字典形式，{0: 115, 1: 80, 2: 40, 3: 135, 4: 108, 5: 153 }

    for i, lable in enumerate(labels[0]):
        if lable != 0:
            break#只输出类别为0的数据
        print(f'{keys[i]}_{lable}') # TwA_jJIBGqEgyLySb35D_0, ids是TwA_jJIBGqEgyLySb35D, 类别是 0
        

def find_all_npz(npz_path):
    npz_all = []
    for dirpath, _, filenames in os.walk(npz_path):
        for filename in filenames:
            if filename.endswith('.npz'):
                npz_all.append(os.path.join(dirpath,filename))

    return npz_all

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Embeddings_Cluster")
    parser.add_argument("--npz_dir", "-nd", type=str, required=True, help="Directory containing NPZ files.")
    parser.add_argument("--clusters_num", "-cn", type=int, required=True, help="")
    parser.add_argument("--iters", "-it",type=int, required=True, default=100, help="")

    args = parser.parse_args()

    embeddings, keys = load_embeddings(args.npz_dir)


    print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")

    n_clusters = args.clusters_num

    cluster_centers, labels, distance = perform_kmeans(embeddings, n_clusters, n_iter=args.iters, gpu=True)

    print(f"Cluster centers shape: {cluster_centers.shape}")
    print(f"Labels shape: {labels.shape}")

    sampled_keys, sampled_labels, label_counts = sample_from_clusters(embeddings, keys, labels, distance, n_samples=5)

    label_keys = np.array(list(label_counts.keys()))
    label_values = np.array(list(label_counts.values()))
    
    np.savez("shuping/cluster_results.npz", 
            #  cluster_centers=cluster_centers, 
             labels=labels,
             distance=distance, 
             keys=keys)
    np.savez("shuping/sampled_results.npz", 
             cluster_centers=cluster_centers, 
             sampled_keys=sampled_keys,
             sampled_labels=sampled_labels,
             label_keys=label_keys, 
             label_values=label_values)

    print("Clustering results saved to 'cluster_results.npz'.")
    print("Sampled embeddings and keys saved to 'sampled_results.npz'.")
    read_from_npz()