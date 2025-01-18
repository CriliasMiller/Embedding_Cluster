import os

def find_all_npz(npz_path):
    npz_all = []
    for dirpath, _, filenames in os.walk(npz_path):
        for filename in filenames:
            if filename.endswith('.tar'):
                npz_all.append(os.path.join(dirpath,filename))

    return npz_all

print(len(find_all_npz('/workspace/Data/video_and_image_datas/processed_videos_wds_datas/hd_movie_video_data2_asia/')))