from functools import partial
import yaml
from data.dataset import Objaverse
from data.data_utils import to_mesh
from data.serializaiton import detokenize
from model.nautilus import MeshTransformer
import torch
import os
import argparse
from torch import is_tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import trimesh
from x_transformers.autoregressive_wrapper import top_p, top_k
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--mesh_path', type=str, default=None)
parser.add_argument('--pc_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default='output')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--temperature', type=float, default=1)
args = parser.parse_args()


def joint_filter(logits, k = 50, p=0.95):
    logits = top_k(logits, k = k)
    logits = top_p(logits, thres = p)
    return logits


def first(it):
    return it[0]


def custom_collate(data, pad_id = -1):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output


if __name__ == '__main__':
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = MeshTransformer(
        dim = config['dim'],
        attn_depth = config['depth'],
        max_seq_len = config['max_seq_len'],
        num_discrete_coors= 2**int(config['quant_bit']),
        u_size= config['u_size'],
        v_size= config['v_size'],
        encoder_name = config['encoder_name'],
    )
    model.load(args.model_path)

    model = model.eval()
    os.makedirs(args.output_path, exist_ok=True)
    
    num_params = sum([param.nelement() for param in model.decoder.parameters()])
    print('Number of parameters: %.2f M' % (num_params / 1e6))

    mesh_dir = args.mesh_path
    pc_dir = args.pc_path
    val_dataset = Objaverse(mesh_dir if mesh_dir is not None else pc_dir, load_pc=True, pc_dir=pc_dir,
                            return_mesh=mesh_dir is not None, u_size=config['u_size'], v_size=config['v_size'],
                            quant_bit=config['quant_bit'])
    dataloader = DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = False,
        collate_fn = partial(custom_collate, pad_id=-1)
    )
    
    # accelerate inference
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        project_dir=args.output_path,
        kwargs_handlers=[kwargs]
    )
    if accelerator.state.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    dataloader, model = accelerator.prepare(dataloader, model)

    with torch.no_grad(), accelerator.autocast():
        model = model.module if accelerator.state.num_processes > 1 else model
        model = model.half()
        accelerator.print(f'U Size: {model.u_size} | V Size: {model.v_size}')
        
        for it, data in enumerate(dataloader):
            codes = model.generate(
                batch_size = args.batch_size,
                temperature = args.temperature,
                pc = data['pc'].half(),
                filter_logits_fn = joint_filter,
                filter_kwargs = dict(k=50, p=0.95),
                return_codes=True,
            )
                
            coords = []
            try:
                for i in range(len(codes)):
                    code = codes[i]
                    code = code[code != model.pad_id].cpu().numpy()
                    vertices = detokenize(
                        code, 
                        u_size= model.u_size,
                        v_size= model.v_size,
                        num_discrete_coors = model.num_discrete_coors
                    )
                    coords.append(vertices)
            except:
                coords.append(np.zeros((3, 3)))
            
            device_idx = accelerator.device.index
            for i in range(args.batch_size):
                uid = data['uid'][i]
                vertices = coords[i]
                faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
                mesh = to_mesh(vertices, faces, transpose=False, post_process=True)
                num_faces = len(mesh.faces)
                face_color = np.array([120, 154, 192, 255], dtype=np.uint8)
                face_colors = np.tile(face_color, (num_faces, 1))
                mesh.visual.face_colors = face_colors
                mesh.export(f'{args.output_path}/{uid}_mesh.obj')

                pcd = data['pc'][i].cpu().numpy()
                point_cloud = trimesh.points.PointCloud(pcd[..., 0:3])
                point_cloud.export(f'{args.output_path}/{uid}_pc.ply', "ply")
            
            print('saved at:', args.output_path)

