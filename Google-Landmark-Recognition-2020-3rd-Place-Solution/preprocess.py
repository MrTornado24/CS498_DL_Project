import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold

project_path = '/mnt/data/sjx/CS498_DL_Project'
method_path = 'Google-Landmark-Recognition-2020-3rd-Place-Solution'


def main():
    df_train = pd.read_csv(os.path.join(project_path, 'data', 'train.csv'))

    skf = StratifiedKFold(5, shuffle=True, random_state=233)

    df_train['fold'] = -1
    for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['landmark_id'])):
        df_train.loc[valid_idx, 'fold'] = i
        
    df_train.to_csv(os.path.join(project_path, 'data', 'train_0.csv'), index=False)

    landmark_id2idx = {idx: landmark_id for idx, landmark_id in enumerate(sorted(df_train['landmark_id'].unique()))}
    with open(os.path.join(project_path, 'idx2landmark_id.pkl'), 'wb') as fp:
        pickle.dump(landmark_id2idx, fp)


if __name__ == '__main__':
    main()
