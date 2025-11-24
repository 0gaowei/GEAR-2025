
import torch

def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores.cpu()
    labels = labels.cpu()
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights).sum(1)
       idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg

    return metrics

def hr(scores, labels, k):
    """计算 Hit Rate@K (HR@K)
    HR@K = 在Top-K推荐中是否命中目标物品的平均值
    """
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    # HR@K: 只要Top-K中有任何一个正样本命中，就记为1
    hr = (hits.sum(1) > 0).float().mean().item()
    return hr


def mrr(scores, labels):
    """计算 Mean Reciprocal Rank (MRR)
    MRR = 平均倒数排名，即第一个正样本排名的倒数
    """
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    
    # 找到每个样本中第一个正样本的位置
    batch_size = labels.size(0)
    reciprocal_ranks = []
    
    for i in range(batch_size):
        # 找到该样本中所有正样本的位置
        positive_indices = labels[i].nonzero(as_tuple=True)[0]
        if len(positive_indices) == 0:
            # 如果没有正样本，MRR为0
            reciprocal_ranks.append(0.0)
        else:
            # 找到第一个正样本在排序后的排名位置（从1开始）
            sorted_indices = rank[i]
            first_positive_rank = None
            for pos_idx in positive_indices:
                # 找到pos_idx在sorted_indices中的位置（从0开始）
                rank_pos = (sorted_indices == pos_idx).nonzero(as_tuple=True)[0]
                if len(rank_pos) > 0:
                    rank_value = rank_pos[0].item() + 1  # 转换为从1开始的排名
                    if first_positive_rank is None or rank_value < first_positive_rank:
                        first_positive_rank = rank_value
            if first_positive_rank is not None:
                reciprocal_ranks.append(1.0 / first_positive_rank)
            else:
                reciprocal_ranks.append(0.0)
    
    return torch.tensor(reciprocal_ranks).float().mean().item()


def compute_metrics(scores, labels):
    """计算 HR@5, NDCG@5, HR@10, NDCG@10, MRR 这5个指标"""
    metrics = {}
    
    scores = scores.cpu()
    labels = labels.cpu()
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    
    # 计算 HR@5
    cut_5 = rank[:, :5]
    hits_5 = labels_float.gather(1, cut_5)
    metrics['HR@5'] = (hits_5.sum(1) > 0).float().mean().item()
    
    # 计算 NDCG@5
    position_5 = torch.arange(2, 2+5)
    weights_5 = 1 / torch.log2(position_5.float())
    dcg_5 = (hits_5 * weights_5).sum(1)
    idcg_5 = torch.Tensor([weights_5[:min(n, 5)].sum() if n > 0 else 1.0 for n in answer_count])
    ndcg_5 = (dcg_5 / idcg_5.clamp(min=1e-8)).mean()
    metrics['NDCG@5'] = ndcg_5.item()
    
    # 计算 HR@10
    cut_10 = rank[:, :10]
    hits_10 = labels_float.gather(1, cut_10)
    metrics['HR@10'] = (hits_10.sum(1) > 0).float().mean().item()
    
    # 计算 NDCG@10
    position_10 = torch.arange(2, 2+10)
    weights_10 = 1 / torch.log2(position_10.float())
    dcg_10 = (hits_10 * weights_10).sum(1)
    idcg_10 = torch.Tensor([weights_10[:min(n, 10)].sum() if n > 0 else 1.0 for n in answer_count])
    ndcg_10 = (dcg_10 / idcg_10.clamp(min=1e-8)).mean()
    metrics['NDCG@10'] = ndcg_10.item()
    
    # 计算 MRR (Mean Reciprocal Rank)
    # 对于每个样本，找到第一个正样本的排名位置
    batch_size = labels.size(0)
    num_items = labels.size(1)
    reciprocal_ranks = torch.zeros(batch_size)
    
    for i in range(batch_size):
        positive_mask = labels[i] > 0
        if positive_mask.any():
            # 找到第一个正样本在排序后的排名位置（从1开始）
            # rank[i] 是排序后的索引，我们需要找到正样本在rank中的位置
            positive_indices = positive_mask.nonzero(as_tuple=True)[0]
            # 找到这些正样本在rank[i]中的位置
            rank_positions = []
            for pos_idx in positive_indices:
                # 找到pos_idx在rank[i]中的位置
                pos_in_rank = (rank[i] == pos_idx).nonzero(as_tuple=True)[0]
                if len(pos_in_rank) > 0:
                    rank_positions.append(pos_in_rank[0].item() + 1)  # 转换为从1开始的排名
            
            if rank_positions:
                first_rank = min(rank_positions)
                reciprocal_ranks[i] = 1.0 / first_rank
            else:
                reciprocal_ranks[i] = 0.0
        else:
            reciprocal_ranks[i] = 0.0
    
    metrics['MRR'] = reciprocal_ranks.mean().item()
    
    return metrics


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]