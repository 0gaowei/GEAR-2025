from .yelp import YelpDataset
from .retail import RetailDataset
from .ijcai import IjcaiDataset
from .taobao import TaobaoDataset

DATASETS = {
    YelpDataset.code(): YelpDataset,
    RetailDataset.code(): RetailDataset,
    IjcaiDataset.code(): IjcaiDataset,
    TaobaoDataset.code(): TaobaoDataset
}


def dataset_factory(
        dataset_code,
        target_behavior,
        multi_behavior,
        min_uc,
        train_file: str = None,
        val_file: str = None,
        test_file: str = None,
    ):
    dataset_cls = DATASETS[dataset_code]
    return dataset_cls(
        target_behavior,
        multi_behavior,
        min_uc,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
    )
