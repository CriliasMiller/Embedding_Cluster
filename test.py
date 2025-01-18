import numpy as np
from tqdm import tqdm
import faiss
# def read_from_npz():
#     npz_file = '/workspace/Crilias/zhangzhenxing/data/clustering/shuping/sampled_results.npz'

#     data = np.load(npz_file)

#     keys = []
#     labels = []
#     label_counts = []

#     keys.append(data['sampled_keys'])
#     labels.append(data['sampled_labels'])

#     keys = np.concatenate(keys)
#     labels = np.array(labels)

#     label_keys = data['label_keys']
#     label_values = data['label_values']
#     label_counts = dict(zip(label_keys, label_values))

#     print("Label counts:", label_counts[0])#字典形式，{0: 115, 1: 80, 2: 40, 3: 135, 4: 108, 5: 153 }

#     label_now = 0
#     for i, label in enumerate(labels[0]):
#         if label!= label_now:
#             print("--------")
#             label_now = label
#         if label > 10 :
#             break
#         if label < 10:
#             print(f'{keys[i]}') # TwA_jJIBGqEgyLySb35D_0, ids是TwA_jJIBGqEgyLySb35D, 类别是 0

# def sample_from_npz(n_samples=5):
#     npz_file = '/workspace/Crilias/zhangzhenxing/data/clustering/cluster_results.npz'

#     data = np.load(npz_file)

#     keys = []
#     labels = []
#     distance = []
#     label_counts = {}
    
#     keys.append(data['keys'])
#     labels.append(data['labels'])
#     # distance.append(data['labels'])

#     keys = np.concatenate(keys)
#     labels = np.array(labels[0])

#     label_counts = {}
#     sampled_keys = []
#     sampled_labels = []
#     unique_labels = np.unique(labels)

#     for label in tqdm(unique_labels):
#         indices = np.where(labels == label)[0]
#         label_counts[label] = len(indices)
#         # sorted_indices = indices[np.argsort(distance[indices])]

#         if len(indices) > n_samples:
#             # sampled_indices = sorted_indices[:n_samples] #最近采样
#             sampled_indices = np.random.choice(indices, n_samples, replace=False) #random 采样
#         else:
#             sampled_indices = indices

#         sampled_keys.append(keys[sampled_indices])
#         sampled_labels.extend([label] * len(sampled_indices))
#     # print(len(sampled_labels))
#     sampled_keys = np.concatenate(sampled_keys)
#     sampled_labels = np.array(sampled_labels)

#     return sampled_keys, sampled_labels, label_counts

# # sampled_keys, sampled_labels, label_counts = sample_from_npz()
# # label_keys = np.array(list(label_counts.keys()))
# # label_values = np.array(list(label_counts.values()))
    
# # np.savez("sampled_New_results.npz", 
# #              sampled_keys=sampled_keys,
# #              sampled_labels=sampled_labels,
# #              label_keys=label_keys, 
# #              label_values=label_values)
# read_from_npz()

