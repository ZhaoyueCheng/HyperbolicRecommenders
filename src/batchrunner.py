import torch
import numpy as np
from tqdm import tqdm
from .metrics import metrics
from collections import defaultdict

def train(
        loader,
        model,
        optimizer,
        criterion,
        scheduler = None,
        masked_loss = False,
        show_progress = True
    ):
    model.train()
    losses = []
    
    if show_progress:
        loader = tqdm(loader)    
    
    for batch in loader:
        optimizer.zero_grad()
        dense_batch = batch.to_dense() # criterion may not support sparse tensor
        predictions = model(dense_batch)
        if masked_loss:
            mask = dense_batch > 0
            loss = criterion(predictions.masked_select(mask), dense_batch.masked_select(mask))
        else:
            loss = criterion(predictions, dense_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    if scheduler is not None:        
        scheduler.step()
    return losses
    

def evaluate(
        loader,
        eval_data,
        model,
        top_k = [10],
        report_coverage = True,
        show_progress = True
    ):
    model.eval()
    
    coverage_set = None
    if report_coverage:
        coverage_set = {k:set() for k in top_k}
        
    if show_progress:
        loader = tqdm(loader)        
    
    results = defaultdict(list)
    for i, batch in enumerate(loader):
        dense_batch = batch.to_dense().squeeze() # single user prediction
        with torch.no_grad():
            predictions = model(dense_batch)
        
        itemid = eval_data['items_data'][i]
        labels = eval_data['label_data'][i]
        predicted = predictions[itemid]
        
        for k in top_k:
            scores = metrics(predicted, labels, top_k=k, coverage_set=coverage_set[k])
            for score, metric in zip(scores, ['hr', 'arhr', 'ndcg']):
                results[f"{metric}@{k}"].append(score)
        
    results = {metric: np.mean(score) for metric, score in results.items()}
    
    if report_coverage:
        for k, cov in coverage_set.items():
            results[f"cov@{k}"] = len(cov)
    
    return results

def report_metrics(scores, epoch=None):
    log_str = f'Epoch: {epoch}' if epoch is not None else 'Scores'
    log = f"{log_str} | " + " | ".join(map(lambda x: f'{x[0]}: {x[1]:.6f}', scores.items()))
    print(log)

def evaluate_new(loader, train_mat, eval_data, model, show_progress=False):
    model.eval()
    if show_progress:
        loader = tqdm(loader)

    pred_matrix = predict_new(loader, train_mat, model)
    precision, recall, ndcg = eval_rec(train_mat, eval_data, pred_matrix)

    print("------------------------")
    print("Recall: ")
    print(recall)
    print("NDCG: ")
    print(ndcg)


def predict_new(loader, train_mat, model):
    num_users, num_items = train_mat.shape[0], train_mat.shape[1]
    probs_matrix = np.zeros((num_users, num_items))

    for i, batch in enumerate(loader):
        dense_batch = batch.to_dense().squeeze()  # single user prediction
        with torch.no_grad():
            predictions = model(dense_batch).detach().cpu().numpy()

        probs_matrix[i] = np.reshape(predictions, [-1, ])

    return probs_matrix

def eval_rec(train_mat, eval_data, pred_matrix):
    topk = 50
    pred_matrix[train_mat.nonzero()] = np.NINF
    ind = np.argpartition(pred_matrix, -topk)
    ind = ind[:, -topk:]
    arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
    pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]

    precision, recall, MAP, ndcg = [], [], [], []

    # ranking = argmax_top_k(pred_matrix, topk)  # Top-K items

    for k in [5, 10, 20]:
        precision.append(precision_at_k(eval_data, pred_list, k))
        recall.append(recall_at_k(eval_data, pred_list, k))
        # MAP.append(mapk(data.test_dict, pred_list, k))

    all_ndcg = ndcg_lgcn([*eval_data.values()], pred_list)
    ndcg = [all_ndcg[x - 1] for x in [5, 10, 20]]

    return precision, recall, ndcg

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(actual)
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)
    return sum_precision / num_users

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users

def ndcg_lgcn(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        # idcg = np.cumsum(1.0/np.log2(np.arange(2, len_rank+2)))
        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)