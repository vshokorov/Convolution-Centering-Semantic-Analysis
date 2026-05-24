import numpy as np
import pandas as pd
import pickle
import codecs
import torch
import random
import argparse
# from utils import telegram_bot_sendtext
import os
import math
import logging

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def BaselineTestProcedure(model_name, embs_dir, data_dir, data_type='black_ms1m', wandb_logger=None, wandb_log_info='', device='cuda', batch_size=1000):
    """
    Parameters
    ----------
    model_name : str - name of model run. The same name as the csv file with embeddings.
    data_type : str - (white_ms1m/black_ms1m/wf42)
    log_to_tg : bool
    wandb_run_path : str - format: entity/project/run_id
    device : str - (cpu/cuda)
    batch_size : int - the size of the batch that is used to count the top_distances. Affects the allocated memory. 
    """

    if device == 'cuda' and not torch.cuda.is_available():
        logging.info('BaselineTestProcedure: WARNING: Cuda is not available!')
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    logging.info('BaselineTestProcedure: Start processing for ', model_name)

    if data_type == 'white_ms1m':
        df = pd.read_csv(os.path.join(embs_dir, 'ms1m', 'embeddings_'+model_name+'.csv'))
    elif data_type == 'black_ms1m':
        df = pd.read_csv(os.path.join(embs_dir, 'query', 'embeddings_'+model_name+'.csv'))
    elif data_type == 'wf42':
        df = pd.read_csv(os.path.join(embs_dir, 'WF42', 'embeddings_'+model_name+'.csv'))
    else:
        raise ValueError()
     
    df['embedding'] = df['embedding'].apply(lambda x: pickle.loads(codecs.decode(x.encode(), 'base64')))

    df_distractors = pd.read_csv(os.path.join(embs_dir, 'distractors', 'embeddings_'+model_name+'.csv'))
    df_distractors['embedding'] = df_distractors['embedding'].apply(lambda x: pickle.loads(codecs.decode(x.encode(), 'base64')))

    with open(os.path.join(data_dir, 'filter_dis_daa-1.txt'), 'r') as fr:
        filter_dis = fr.readlines()
    filter_dis = [x.strip() for x in filter_dis]

    num = 0
    df_distractors_filtered = []
    for i in range(len(df_distractors)):
        if df_distractors['image'][i] in filter_dis:
            num += 1
        else:
            df_distractors_filtered.append([df_distractors['image'][i], df_distractors['embedding'][i]])

    df_distractors_filtered = pd.DataFrame(df_distractors_filtered, index=None)
    df_distractors_filtered.columns = ['image', 'embedding']
    # Количество изображений по датасетам:
    # 
    # megaface            36249
    # 
    # deepglint           84893
    # 
    # img_align_celeba     9347

    with open(os.path.join(data_dir, 'filter_query.txt'), 'r') as fr:
        query_filter_set = fr.readlines()
    query_filter_set = [x.strip() for x in query_filter_set]


    # ### Процедура тестирования

    # Далее приведены пункты напрямую из инструкции по процедуре тестирования с соответствующим кодом.
    # 
    # 1) К каждой картинке из query подбирается 100 ближайших картинок из distractors. Для каждой картинки q \in query получается 100 пар (q, d). Всего 100|query| пар. Пары сортируются по возрастанию similarity (cosine distance)

    # Комментарий:
    # 
    # Тут создаются 2 тензора sorted_top_distances и closest. sorted_top_distances - пары из 1 пункта по инструкции выше, closest - тензор ближайших к конкретному человеку из query дистракторов - будет использоваться дальше при расчете TPR.
    # 
    n_distractors = 100
    query = torch.from_numpy(np.stack(df.embedding.values)).to(device)
    distractors_filtered = torch.from_numpy(np.stack(df_distractors_filtered.embedding.values))

    # Считаем размер каждого батча
    num_samples = distractors_filtered.shape[0] 
    batch_size = min(n_distractors, batch_size)
    num_steps = math.ceil(num_samples / batch_size)
    
    # Инициализируем массив
    top_distances = - torch.ones(query.shape[0], n_distractors).to(device)
    
    for i in range(num_steps):
        left_range = i * batch_size
        right_range = min((i + 1) * batch_size, num_samples)
        
        # Считаем расстояние от всех query-изображений до всех дистракторов
        distractors_batch = distractors_filtered[left_range:right_range].to(device)
        distances = torch.matmul(query, distractors_batch.transpose(0, 1))
        # Сортируем эти расстояния
        sorted_distances = torch.topk(distances, k=min(n_distractors, len(distractors_batch)), dim=1, largest=True).values

        # Берём 100 ближайших дистракторов для каждого query-изображения, складываем их в один тензор и снова сортируем
        top_distances = torch.cat((top_distances, sorted_distances), dim=1)
        top_distances = torch.topk(top_distances, k=n_distractors, dim=1, largest=True).values

    # Запоминаем ближайших дистракторов - будут использоваться дальше в процессе теста
    closest = top_distances[:, 0]
    top_distances = top_distances.flatten()
    sorted_top_distances = torch.sort(top_distances, descending=True).values

    # 2) Для каждой картинки из query вычисляется similarity между ней и всеми картинками из query, соответствующими тому же id (кроме нее же самой). То есть, если в query для каждого id ровно по 5 картинок, то количество пар будет 4*|query|. 

    df['person'] = df['image'].apply(lambda x: x.split('/')[0])
    df['id'] = df.index

    similarities = list()
    num_pos_pairs = 0

    # Итерируемся по query-датасету
    df_filtered = df[~df['image'].isin(query_filter_set)]
    grouped = df_filtered.groupby('person')
    for _, group in grouped:
        for _, row in group.iterrows():
            emb = row['embedding']
            
            # Берём остальные изображения этого человека
            # Для них считаем similarity
            for _, row_2 in group.iterrows():
                if row['image'] != row_2['image']:
                    similarities.append((row['image'], np.dot(row_2['embedding'], emb)))
                    num_pos_pairs += 1

    # 3) Каждой пары (p1, p2) из пункта 2:
    # Если sim(p1, p2) > sim(p1, d), где d – ближайший к p1 дистрактор И converted_sim(p1, p2) > thr, то pos_sim += 1.

    closest_dict = {img: sim for img, sim in zip(df['image'].values, closest)}
    pos_num = 0
    for i in similarities:
        if i[1] >= closest_dict[i[0]]:
            pos_num += 1


    # 4) Итоговый TPR = pos_num / total_num, где total_num – общее количество пар из пункта 2.

    err_rates = [1e-8, 1e-7, 1e-6, 1e-5]
    tprs = list()

    # 5) Для заданного fpr вычисляется допустимое количество false positive пар (q, d):
    # Fp_num = int(round(total_num * err_rate)),
    # Где total_num – |distractors|*|query|. 
    for err_rate in err_rates:
        fp_num = max([1, round(len(df)*len(df_distractors)*err_rate)])
        # 6) Берутся первые fp_num пар из отсортированного списка из пункта 1. Значение similarity пары под номером fp_num будет трешхолдом (thr).
        thr = sorted_top_distances[fp_num]

        pos_num = 0
        for i in similarities:
            if i[1] >= thr and i[1] >= closest_dict[i[0]]:
                pos_num += 1

        tprs.append(pos_num/num_pos_pairs)

    # Для подсчета top1:
    pos_num = 0
    for i in similarities:
        if i[1] >= closest_dict[i[0]]:
            pos_num += 1

    message = "BaselineTest ready for `{}`".format(model_name)
    if data_type == 'white_ms1m':
        message += '\n*on train data* (Asia, White)\n'
    elif data_type == 'black_ms1m':
        message += '\n(black)\n'
    elif data_type == 'wf42':
        message += '\n(WF42)\n'
    message += '```\n|'
    for i in [8, 7, 6, 5]:
        message += '{message:{fill}{align}{width}}'.format(
           message='TPR@1e-' + str(i),
           fill=' ',
           align='^',
           width=10,
        )
        message += '|'
    message += '\n|'
    for i in tprs:
        message += '{message:{fill}{align}{width}}'.format(
           message= '{:.2f}'.format(i*100),
           fill=' ',
           align='^',
           width=10,
        )
        message += '|'
    message += '\ntop1: {:.2%}```'.format(pos_num/num_pos_pairs)
    
    logging.info(message)
    
    if wandb_logger is not None:
        
        if wandb_log_info != '':
            wandb_log_info = '_' + wandb_log_info
        
        for i, t in zip(range(8, 4, -1), tprs):
            wandb_logger.summary['TPR@1e-' + str(i) + wandb_log_info] = t * 100
        wandb_logger.summary['top1' + wandb_log_info] = pos_num/num_pos_pairs

    torch.cuda.empty_cache()
    return tprs + [pos_num/num_pos_pairs]