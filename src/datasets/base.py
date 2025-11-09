RAW_DATASET_ROOT_FOLDER = 'data'

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import pickle
import hashlib
from typing import Optional, Dict, List
import numpy as np

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self,
            target_behavior,
            multi_behavior,
            min_uc,
            train_file: Optional[str] = None,
            val_file: Optional[str] = None,
            test_file: Optional[str] = None
        ):
        self.target_behavior = target_behavior
        self.multi_behavior = multi_behavior
        self.min_uc = min_uc
        self.bmap = None
        assert self.min_uc >= 2, 'Need at least 2 items per user for validation and test'
        self.split = 'leave_one_out'
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self._external_cache_dir = None
        self._use_external_splits = any([train_file, val_file, test_file])

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @abstractmethod
    def load_df(self):
        pass

    def load_dataset(self):
        if self._use_external_splits:
            return self._load_external_split_dataset()
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        if self._use_external_splits:
            return
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        df = self.load_df()
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        
        df['time_gap'] = df.groupby('uid')['timestamp'].diff().fillna(0)
        
        df, umap, smap, bmap = self.densify_index(df)
        self.bmap = bmap

        train, train_b, val, val_b, train_t, val_t, val_num = self.split_df(df, len(umap))
        dataset = {
            'train': train,
            'val': val,
            'train_b': train_b,
            'val_b': val_b,
            'train_t': train_t, 
            'val_t': val_t, 
            'val_num': val_num,
            'umap': umap,
            'smap': smap,
            'bmap': bmap
        }
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def make_implicit(self, df):
        print('Behavior selection')
        if self.multi_behavior:
            pass
        else:
            df = df[df['behavior'] == self.target_behavior]
        return df

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: (i+1) for i, u in enumerate(set(df['uid']))}
        smap = {s: (i+1) for i, s in enumerate(set(df['sid']))}
        bmap = {b: (i+1) for i, b in enumerate(set(df['behavior']))}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        df['behavior'] = df['behavior'].map(bmap)
        return df, umap, smap, bmap
    
    # def densify_index(self, df):
    #     print('Densifying index')
    #     umap = {u: u for u in set(df['uid'])}
    #     smap = {s: s for s in set(df['sid'])}
    #     bmap = {'pv': 1, 'fav':2, 'cart':3, 'buy':4} if 'buy' in set(df['behavior']) else {'tip': 1, 'neg':2, 'neutral':3, 'pos':4}
    #     df['behavior'] = df['behavior'].map(bmap)
    #     return df, umap, smap, bmap

    def split_df(self, df, user_count):      
        if self.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d['sid']))
            user2behaviors = user_group.progress_apply(lambda d: list(d['behavior']))

            process_time_gap = 'time_gap' in df.columns
            if process_time_gap:
                user2timegaps = user_group.progress_apply(lambda d: list(d['time_gap']))
            else:
                user2timegaps = None
            
            train, train_b, val, val_b = {}, {}, {}, {}
            train_t, val_t = ({}, {}) if process_time_gap else (None, None)
            
            for user in range(1, user_count + 1):
                items = user2items[user]
                behaviors = user2behaviors[user]
                timegaps = user2timegaps[user] if process_time_gap else []
                
                if behaviors[-1] == self.bmap[self.target_behavior]:
                    train[user], val[user] = items[:-1], items[-1:]
                    train_b[user], val_b[user] = behaviors[:-1], behaviors[-1:]
                    if process_time_gap:
                        train_t[user], val_t[user] = timegaps[:-1], timegaps[-1:]
                else:
                    train[user] = items
                    train_b[user] = behaviors
                    if process_time_gap:
                        train_t[user] = timegaps
            
            if process_time_gap:
                return train, train_b, val, val_b, train_t, val_t, len(val)
            else:
                return train, train_b, val, val_b, None, None, len(val)
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}-min_uc{}-target_B{}_MB{}-split{}' \
            .format(self.code(), self.min_uc, self.target_behavior, self.multi_behavior, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

    def get_cache_dir(self) -> Path:
        if not self._use_external_splits:
            cache_dir = self._get_preprocessed_folder_path()
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir
        if self._external_cache_dir is None:
            parts: List[str] = []
            for p in [self.train_file, self.val_file, self.test_file]:
                if p:
                    parts.append(str(Path(p).resolve()))
            joined = '|'.join(parts)
            cache_hash = hashlib.md5(joined.encode('utf-8')).hexdigest()[:8]
            base_dir = Path(self.train_file).resolve().parent if self.train_file else Path('.').resolve()
            cache_dir = base_dir / f'external_cache_{self.code()}_{cache_hash}'
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._external_cache_dir = cache_dir
        return self._external_cache_dir

    def _read_split_file(self, file_path: Optional[str]) -> Optional[pd.DataFrame]:
        if file_path is None:
            return None
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f'Split file not found: {file_path}')
        df = pd.read_csv(path, sep='\t')
        df.columns = [c.split(':')[0] for c in df.columns]
        if 'user_id' not in df.columns or 'item_id' not in df.columns:
            raise ValueError(f'RecBole split file {file_path} must contain user_id and item_id columns')
        return df

    def _load_external_split_dataset(self) -> Dict:
        train_df = self._read_split_file(self.train_file)
        if train_df is None:
            raise ValueError('train_file must be provided when using external splits')
        val_df = self._read_split_file(self.val_file)
        test_df = self._read_split_file(self.test_file)

        combined_frames = [df for df in [train_df, val_df, test_df] if df is not None]
        combined_df = pd.concat(combined_frames, ignore_index=True)
        combined_df['user_id'] = combined_df['user_id'].astype(str)
        combined_df['item_id'] = combined_df['item_id'].astype(str)

        timestamp_col = 'timestamp' if 'timestamp' in combined_df.columns else None
        behavior_col = None
        for col in ['behavior', 'action_type', 'inter_behaviors']:
            if col in combined_df.columns:
                behavior_col = col
                break
        if behavior_col is None:
            combined_df['behavior'] = self.target_behavior
            behavior_col = 'behavior'

        unique_users = sorted(combined_df['user_id'].unique())
        unique_items = sorted(combined_df['item_id'].unique())
        unique_behaviors = sorted(combined_df[behavior_col].unique())

        umap = {u: idx + 1 for idx, u in enumerate(unique_users)}
        smap = {s: idx + 1 for idx, s in enumerate(unique_items)}
        bmap = {b: idx + 1 for idx, b in enumerate(unique_behaviors)}
        self.bmap = bmap

        def prepare_split(df: Optional[pd.DataFrame]):
            if df is None or df.empty:
                return {}, {}, {}, {}
            local_df = df.copy()
            local_df['user_id'] = local_df['user_id'].astype(str)
            local_df['item_id'] = local_df['item_id'].astype(str)
            local_df['uid'] = local_df['user_id'].map(umap)
            local_df['sid'] = local_df['item_id'].map(smap)
            local_df['behavior_code'] = local_df[behavior_col].map(bmap)
            if timestamp_col:
                local_df['timestamp'] = local_df[timestamp_col].astype(float)
                sort_cols = ['uid', 'timestamp']
            else:
                local_df['timestamp'] = 0.0
                sort_cols = ['uid']
            local_df = local_df.sort_values(sort_cols)
            grouped = local_df.groupby('uid')
            items: Dict[int, List[int]] = {}
            behaviors: Dict[int, List[int]] = {}
            timestamps: Dict[int, List[float]] = {}
            last_timestamp: Dict[int, Optional[float]] = {}
            for uid, group in grouped:
                items[uid] = group['sid'].tolist()
                behaviors[uid] = group['behavior_code'].tolist()
                ts_list = group['timestamp'].tolist()
                timestamps[uid] = ts_list
                last_timestamp[uid] = ts_list[-1] if ts_list else None
            return items, behaviors, timestamps, last_timestamp

        train_items, train_behaviors, train_timestamps, train_last_ts = prepare_split(train_df)
        val_items, val_behaviors, val_timestamps, val_last_ts = prepare_split(val_df)
        test_items, test_behaviors, test_timestamps, _ = prepare_split(test_df)

        def compute_time_gaps(timestamp_dict: Dict[int, List[float]], reference_times: Optional[Dict[int, Optional[float]]] = None):
            gaps: Dict[int, List[float]] = {}
            for uid, ts_list in timestamp_dict.items():
                if not ts_list:
                    gaps[uid] = []
                    continue
                prev_time = reference_times.get(uid) if reference_times else None
                current_gaps: List[float] = []
                for ts in ts_list:
                    if prev_time is None:
                        current_gaps.append(0.0)
                    else:
                        current_gaps.append(float(ts) - float(prev_time))
                    prev_time = ts
                gaps[uid] = current_gaps
            return gaps

        train_time_gaps = compute_time_gaps(train_timestamps)
        val_time_gaps = compute_time_gaps(val_timestamps, reference_times=train_last_ts)
        reference_for_test: Dict[int, Optional[float]] = {}
        for uid in test_timestamps.keys():
            ref = None
            if val_last_ts and uid in val_last_ts and val_last_ts[uid] is not None:
                ref = val_last_ts[uid]
            else:
                ref = train_last_ts.get(uid)
            reference_for_test[uid] = ref
        test_time_gaps = compute_time_gaps(test_timestamps, reference_times=reference_for_test)

        for uid in val_items.keys():
            train_items.setdefault(uid, [])
            train_behaviors.setdefault(uid, [])
            train_time_gaps.setdefault(uid, [])
        for uid in test_items.keys():
            train_items.setdefault(uid, [])
            train_behaviors.setdefault(uid, [])
            train_time_gaps.setdefault(uid, [])

        dataset = {
            'train': train_items,
            'val': val_items,
            'train_b': train_behaviors,
            'val_b': val_behaviors,
            'train_t': train_time_gaps,
            'val_t': val_time_gaps,
            'val_num': len(val_items),
            'umap': umap,
            'smap': smap,
            'bmap': bmap,
        }

        if test_items:
            dataset['test'] = test_items
            dataset['test_b'] = test_behaviors
            dataset['test_t'] = test_time_gaps
            dataset['test_num'] = len(test_items)

        if not dataset['train_t']:
            dataset['train_t'] = {uid: [0.0] * len(seq) for uid, seq in train_items.items()}
        if not dataset['val_t']:
            dataset['val_t'] = {uid: [0.0] * len(seq) for uid, seq in val_items.items()}
        if test_items and not dataset.get('test_t'):
            dataset['test_t'] = {uid: [0.0] * len(seq) for uid, seq in test_items.items()}

        return dataset
