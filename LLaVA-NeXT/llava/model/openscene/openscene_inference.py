from torch.utils import model_zoo
from llava.model.openscene.models.disnet import DisNet
import yaml 
from llava.model.openscene.util.config import CfgNode
import os
import torch
from llava.model.openscene.dataset.point_loader import AlignerPoint3DLoader, collation_fn_eval_all
from tqdm import tqdm
from MinkowskiEngine import SparseTensor

def load_cfg_from_cfg_file(file):
    '''Load from config files.'''

    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, value in cfg_from_file[key].items():
            cfg[k] = value

    cfg = CfgNode(cfg)
    return cfg




# cfg = load_cfg_from_cfg_file('/scratch/bczf/zoezheng126/4dllm/LLaVA-NeXT/llava/model/openscene/config/scannet/ours_openseg_pretrained.yaml')

# # load dataset

# cfg.voxel_size = 0.12
# val_data = AlignerPoint3DLoader(datapath_prefix=cfg.data_root,
#                             voxel_size=cfg.voxel_size, 
#                             split=cfg.split, aug=False,
#                             memcache_init=cfg.use_shm, eval_all=False, identifier=6797,
#                             input_color=cfg.input_color)
# val_sampler = None
# cfg.test_batch_size = 1
# val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.test_batch_size,
#                                             shuffle=False, num_workers=cfg.test_workers, pin_memory=True,
#                                             drop_last=False, collate_fn=collation_fn_eval_all,
#                                             sampler=val_sampler)


# # # load model
# model = DisNet(cfg)
# model = model.to('cuda')
# checkpoint = model_zoo.load_url(cfg.model_path, progress=True)
# model.load_state_dict(checkpoint['state_dict'], strict=True)
# model.eval()


# # inference
# for i, (coords, feats, labels, inds_reverse) in enumerate(tqdm(val_data_loader)):
#     print("coords.shape: ", coords.shape, "feats.shape: ", feats.shape, "labels.shape: ", labels.shape, "inds_reverse.shape: ", inds_reverse.shape)
    
#     sinput = SparseTensor(feats.cuda(non_blocking=True), coords.cuda(non_blocking=True))
#     print(f'sinput: {sinput.shape}')
#     coords = coords[inds_reverse, :]
#     pcl = coords[:, 1:].cpu().numpy()
#     predictions, xyz = model(sinput)
#     print("predictions.shape: ", predictions.shape)
#     predictions = predictions[inds_reverse, :]
#     print("predictions.shape after inds_reverse: ", predictions.shape)
#     break
#     # pred = predictions.half() @ text_features.t()
#     # logits_pred = torch.max(pred, 1)[1].cpu()











