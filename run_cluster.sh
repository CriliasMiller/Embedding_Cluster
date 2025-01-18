source /root/miniconda3/etc/profile.d/conda.sh; 
conda activate base;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python kmean.py --clusters_num 10000 --iters 300 --npz_dir /workspace/intern/wangzehui/meta_manage/cluster_analysis/processed_npz_vertical
