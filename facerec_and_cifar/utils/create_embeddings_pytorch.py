#import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
import pandas as pd
import os
from tqdm import trange, tqdm
import time
import codecs
import pickle
import numpy as np
from torchvision import transforms as trans
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch
import random
from skimage.io import imread
import io
import pickle
import zipfile
import pathlib

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class ImgDataset(Dataset):
    def __init__(self, root_path, target_transform, names=None, cache_into_memory=False):
        
        if names is None:
            self.name_list = sorted(list(filter(lambda x: x[-4:] == '.jpg', os.listdir(root_path))))
        else:
            self.name_list = names
        
        self.transform = target_transform
        self.root = root_path

    def __getitem__(self, key):
        img = self.transform(Image.open(os.path.join(self.root, self.name_list[key])).resize((112, 112)))
        return img, key

    def __len__(self):
        return len(self.name_list)
    
class ZipDataset(Dataset):
    def __init__(self, root_path, target_transform, names=None, cache_into_memory=False):
        if cache_into_memory:
            f = open(str(root_path) + '.zip', 'rb')
            self.zip_content = f.read()
            f.close()
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')
        else:
            self.zip_file = zipfile.ZipFile(str(root_path) + '.zip', 'r')
        
        if names is None:
            self.name_list = sorted(list(filter(lambda x: x[-4:] == '.jpg', self.zip_file.namelist())))
        else:
            self.name_list = names
        
        self.transform = target_transform
        self.root = root_path

    def __getitem__(self, key):
        buf = self.zip_file.read(name=self.name_list[key])
        img = self.transform(Image.open(io.BytesIO(buf)))
        return img, key

    def __len__(self):
        return len(self.name_list)
    
def get_embeddings_for_ds(conf,
                          model_forward,
                          ds_class,
                          target_list, 
                          target_list_ds,
                          ds_zip):
    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    ds = ds_class(ds_zip, 
                    transform,
                    target_list_ds,
                    cache_into_memory=conf.cache_zip_data)

    dl = DataLoader(ds, 
                    batch_size=conf.batch_size, 
                    shuffle=False, 
                    pin_memory=True,
                    num_workers=conf.num_workers)

    df = pd.DataFrame(columns=['image', 'embedding'])
    for imgs, labels in tqdm(dl):
        imgs = imgs.cuda()
        labels = labels.numpy()

        with torch.no_grad():
            embeddings = model_forward(imgs).cpu().detach().numpy()

        for lab, emb in zip(labels, embeddings):
            code = codecs.encode(pickle.dumps(emb), "base64").decode()
            name = target_list[lab]
#             df = df.append({'image': name, 'embedding': code}, ignore_index=True)
            df = pd.concat([df, pd.DataFrame.from_records([{'image': name, 'embedding': code}])])
    
    del ds, dl
    return df.reset_index(drop=True)

def create_embeddings(conf, model_forward, query=True, ms1m=True, WF42=False):
    assert any((query, ms1m, WF42))
    
    if not os.path.exists(os.path.join(conf.output, 'embeddings')):
        os.mkdir(os.path.join(conf.output, 'embeddings'))
    if query and not os.path.exists(os.path.join(conf.output, 'embeddings', 'query')):
        os.mkdir(os.path.join(conf.output, 'embeddings', 'query'))
    if ms1m and not os.path.exists(os.path.join(conf.output, 'embeddings', 'ms1m')):
        os.mkdir(os.path.join(conf.output, 'embeddings', 'ms1m'))
    if WF42 and not os.path.exists(os.path.join(conf.output, 'embeddings', 'WF42')):
        os.mkdir(os.path.join(conf.output, 'embeddings', 'WF42'))
    if not os.path.exists(os.path.join(conf.output, 'embeddings', 'distractors')):
        os.mkdir(os.path.join(conf.output, 'embeddings', 'distractors'))

    if query:
        print('Start processing for ', conf.run_name)

        # ## Create query embeddings
        # Списки нужных нам изображений
        print('Embeddings for African (query)')
        with open(os.path.join(conf.rec, 'query_list.txt')) as fr:
            query_list = list(map(lambda x: x.split('\n')[0], 
                                fr.readlines()))
        ds_query_list = ['African/' + x for x in query_list]

        df = get_embeddings_for_ds(
            conf,
            model_forward,
            ZipDataset,
            query_list, 
            ds_query_list,
            os.path.join(conf.rec, 'aligned', 'RFW', 'rfw_african')
        )
        df.to_csv(os.path.join(conf.output, 'embeddings', 'query', 'embeddings_'+conf.run_name+'.csv'), index=False)
    else:
        print('Skip embeddings for African (query)')
    
    if ms1m:
        # ## Create train 3000 classes and 10% images embeddings
        print('Embeddings for source train (ms1m)')

        with open(os.path.join(conf.rec, "tpr@fpr_names_3000.txt"), "rb") as fp:   # Unpickling
            image_names = pickle.load(fp)

        ds_image_names = ['imgs/' + x for x in image_names]

        df = get_embeddings_for_ds(
            conf,
            model_forward,
            ZipDataset,
            image_names, 
            ds_image_names,
            os.path.join(conf.rec, 'imgs')
        )
        df = df.drop_duplicates(subset=['image'])
        df.to_csv(os.path.join(conf.output, 'embeddings', 'ms1m', 'embeddings_'+conf.run_name+'.csv'), index=False)
        
        # ## Create train 3000 classes and 10% images embeddings
        with open(os.path.join(conf.rec, "tpr@fpr_WF42_names_3000.txt"), "rb") as fp:   # Unpickling
            image_names = pickle.load(fp)
        ds_image_names = ['workspace/WebFace42M/' + x for x in image_names]
    else:
        print('Skip embeddings for source train (ms1m)')
    
    if WF42:
        print('Embeddings for source train (WF42)')
        df = get_embeddings_for_ds(
            conf,
            model_forward,
            ZipDataset,
            image_names, 
            ds_image_names,
            os.path.join(conf.rec, 'WebFace42M_3m')
        )
        df = df.drop_duplicates(subset=['image'])
        df.to_csv(os.path.join(conf.output, 'embeddings', 'WF42', 'embeddings_'+conf.run_name+'.csv'), index=False)
    else:
        print('Skip embeddings for source train (WF42)')


    # ## Create distractors embeddings
    with open(os.path.join(conf.rec, 'dis_list.txt'), 'r') as fr:
        distr_list = list(map(lambda x: x.split('\n')[0], fr.readlines()))
    
    print('Embeddings for CelebA')
    image_names = ['img_align_celeba/' + x[x.find('/') + 1:] for x in distr_list if 'celebA' in x]
    ds_image_names = [x for x in image_names]

    df_distractors = get_embeddings_for_ds(
        conf,
        model_forward,
        ImgDataset,
        image_names, 
        ds_image_names,
        os.path.join(conf.rec, 'aligned', 'CelebA/')
    )
    assert df_distractors['image'].apply(lambda x: x.split('/')[0]).value_counts()['img_align_celeba'] == 9347
    
    print('Embeddings for Megaface')
    image_names = ['/'.join(x.split('/')[2:]) for x in distr_list if 'megaface' in x]
    ds_image_names = [x for x in image_names]
    image_names = ['megaface/' + x for x in image_names]

    df_distractors = pd.concat([
        df_distractors, 
        get_embeddings_for_ds(
            conf,
            model_forward,
            ImgDataset,
            image_names,
            ds_image_names,
            os.path.join(conf.rec, 'aligned', 'MegaFace/')
        )],
        ignore_index=True
    )
    assert df_distractors['image'].apply(lambda x: x.split('/')[0]).value_counts()['megaface'] == 36249
    
    print('Embeddings for DeepGlint')
    base = os.path.join(conf.rec, 'aligned', 'DeepGlint/')
    image_names = [x.split('/')[1] for x in distr_list if 'deepglint' in x]
    ds_image_names = []
    for subdir in os.listdir(base):
        if '.txt' in subdir:
            continue
        for name in os.listdir(os.path.join(base, subdir)):
            if name in image_names:
                ds_image_names.append(os.path.join(subdir, name))
    image_names = ['deepglint/' + x for x in image_names]

    df_distractors = pd.concat([
        df_distractors, 
        get_embeddings_for_ds(
            conf,
            model_forward,
            ImgDataset,
            image_names,
            ds_image_names,
            base
        )],
        ignore_index=True
    )
    assert df_distractors['image'].apply(lambda x: x.split('/')[0]).value_counts()['deepglint'] == 84893
    df_distractors.to_csv(os.path.join(conf.output, 'embeddings', 'distractors', 'embeddings_'+conf.run_name+'.csv'), index=False)
    
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    from config import get_config
    from Learner import face_learner

    parser = argparse.ArgumentParser(description='Create embeddings for train data and 3 distractors datasets.')
    parser.add_argument("conf.run_name", type=str)
    parser.add_argument('--data_path', type=str, default=FILE_PATH/'datasets')
    parser.add_argument('--work_space', type=str, default=FILE_PATH/'work_space')
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--model_architecture', type=str, default='ResNet50')
    parser.add_argument('--model_weights', type=str, default='')
    
    args = parser.parse_args()
    conf = get_config(data_path=args.data_path, work_path= args.work_space, training=False)

    conf.data_mode = 'query'
    conf.use_mobilfacenet = False
    conf.model_architecture = args.model_architecture
    conf.device = torch.device("cuda:" + args.gpu_device if torch.cuda.is_available() else "cpu")
    
    conf.data_path = './datasets/'
    conf.embedding_size = 1000
    learner = face_learner(conf, True)
    conf.embedding_size = 512
    
    raise NotImplementedError('You must add weights loading into model')
    
    
    try:
        learner.model.load_state_dict(
            torch.load(args.model_weights, 
                       map_location=conf.device), 
            strict=False)
        
        run_model = learner.run_model
    except:
        import sys
        sys.path.append('/workspace/insightface_mipt/recognition/arcface_torch')
        
        from backbones import get_model
        from torch.nn.functional import normalize
        
        params = dict()
        if '_v2' in args.model_architecture:
            raise NotImplementedError()
            # params.update({'use_se': cfg.use_se,
            #                'drop_block_params': {'keep_prob': cfg.keep_prob,
            #                                      'block_size': cfg.block_size} if cfg.use_dropblock else None})
        
        backbone = get_model(
            args.model_architecture, dropout=0.0, fp16=True, num_features=conf.embedding_size,
            pretrained=False, **params).to(conf.device)
        
        backbone_path = os.path.join(repo_model['path'], callback_checkpoint.backbone_name)
        backbone.load_state_dict(torch.load(backbone_path, map_location=device))
        logging.info(f"Backbone resume successfully (path: {model_line[-1]['path']})!")
        
        
        def run_model(x):
            emb = backbone(x)
            norm_emb = normalize(emb)
            return norm_emb
    
    create_embeddings(args.conf.run_name, conf, run_model)

