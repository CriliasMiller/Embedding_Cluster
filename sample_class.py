import numpy as np
npz_file = '/workspace/Crilias/zhangzhenxing/data/clustering/sampled_results.npz'

data = np.load(npz_file)

embeddings = []
keys = []
lables = []
embeddings.append(data['sampled_embeddings'])
keys.append(data['sampled_keys'])
lables.append(data['sampled_labels'])

print(embeddings)
for i, lable in enumerate(lables):
    print(lable)
    break
    if lable == 0:
        print(keys[i])