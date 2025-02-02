import random
import numpy as np
import torch
from PIL import Image
import json
import os
from os import path
from pathlib import Path
from typing import Dict
from .tokenizer import coordinates_compression, serialize_mesh
from .data_utils import load_process_mesh, to_mesh, jitter_vertices
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
SYNSET_DICT_DIR = Path(__file__).resolve().parent

class Objaverse(torch.utils.data.Dataset):  # pragma: no cover
    def __init__(
        self,
        data_dir: str,
        patch_size: int = 1,
        max_len: int = 9600,
        quant_bit: int = 7,
        augment: bool = False,
        augment_dict: dict = None,
        load_pc: bool = False,
        pc_dir: str = None,
        pc_embeds_dir: str = None,
        pc_num: int = 4096,
        return_mesh: bool = True,
        return_pc: bool = False,
        return_model_path: bool = False,
        u_size = 1024,
        v_size = 2048,
        min_face_num = 5,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.max_len = max_len
        self.quant_bit = quant_bit
        self.augment = augment
        self.augment_dict = augment_dict
        self.load_pc = load_pc
        self.pc_dir = pc_dir
        self.pc_embeds_dir = pc_embeds_dir
        self.pc_num = pc_num
        self.return_mesh = return_mesh
        self.return_pc = return_pc
        self.return_model_path = return_model_path
        self.u_size = u_size
        self.v_size = v_size
        self.min_face_num = min_face_num

        if path.isdir(self.data_dir):
            self.data_path = os.listdir(data_dir)
        elif self.data_dir.endswith('.txt'):
            with open(self.data_dir, 'r') as f:
                self.data_path = [line.strip() for line in f]
            self.data_dir = '/'
        elif self.data_dir.endswith('.json'):
            with open(self.data_dir, 'r') as f:
                data = json.load(f)
            self.data_path = [item['mesh_path'] for item in data]
            self.data_dir = '/'
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data_path)

    def load_mesh(self, model_path, transformation=None):
        # load and process mesh
        mesh = load_process_mesh(model_path, quantization_bits=self.quant_bit, transformation=transformation,
                                 augment=self.augment, augment_dict=self.augment_dict, mc=self.mc)
        
        verts, faces = mesh['vertices'], mesh['faces']
        verts = torch.tensor(verts)
        faces = torch.tensor(faces)
        return verts, faces

    def sample_pc(self, mesh, pc_num_total=50000):
        points, face_idx = mesh.sample(pc_num_total, return_index=True)
        normals = mesh.face_normals[face_idx]
        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

        # random sample point cloud
        ind = np.random.choice(pc_normal.shape[0], self.pc_num, replace=False)
        pc_normal = pc_normal[ind]
        return pc_normal

    def __getitem__(self, idx: int) -> Dict:
        model_path = path.join(self.data_dir, self.data_path[idx])
        model_id = path.basename(self.data_path[idx]).split('.')[0]
        data = {}
        data["uid"] = model_id

        if self.return_mesh:
            for _ in range(5):
                try:
                    verts, faces = self.load_mesh(model_path)
                    break
                except Exception as e:
                    print('Load mesh fail at:', model_path, e)
                    model_path = path.join(self.data_dir, random.choice(self.data_path))
                    model_id = path.basename(model_path).split('.')[0]

            mesh = to_mesh(vertices=verts, faces=faces, transpose=True)
            sequence = serialize_mesh(mesh, special_token=-2)
            codes = coordinates_compression(
                sequence, u_size=self.u_size, v_size=self.v_size, quant_bit=self.quant_bit
            )

            data["vertices"] = verts
            data["faces"] = faces
            data['codes'] = torch.tensor(codes)

            if len(data['codes']) >= self.max_len or len(data['faces']) < self.min_face_num:
                return self.__getitem__(random.randint(0, len(self.data_path) - 1))
            
        if self.return_model_path:
            data['model_path'] = model_path
        
        if self.load_pc:
            if self.pc_embeds_dir is not None:
                pc_embeds_path = os.path.join(self.pc_embeds_dir, model_id + '.pt')
                pc_embeds = torch.load(pc_embeds_path)
                data['pc_embeds'] = pc_embeds.float()
            elif self.pc_dir is not None:
                try:
                    pc_path = os.path.join(self.pc_dir, model_id + '.npz')
                    pc_normal = np.load(pc_path, allow_pickle=True)['data']
                    # random sample point cloud
                    ind = np.random.choice(pc_normal.shape[0], self.pc_num, replace=False)
                    pc_normal = pc_normal[ind]
                    data['pc'] = torch.tensor(pc_normal)

                except Exception as e:
                    print('[Error]: convert point cloud fail at:', model_path, e)
                    jitter_verts = jitter_vertices(verts.clone())
                    mesh = to_mesh(vertices=jitter_verts, faces=faces, transpose=True)
                    pc_normal = self.sample_pc(mesh)
                    data['pc'] = torch.tensor(pc_normal)
            
            else:
                try:
                    # assert len(self.data_path) > 1e5, 'only support training here'
                    # jitter_verts = jitter_vertices(verts.clone())
                    # jitter_mesh = to_mesh(vertices=jitter_verts, faces=faces, transpose=False)
                    pc_normal = self.sample_pc(mesh)
                    data['pc'] = torch.tensor(pc_normal)
                except Exception as e:
                    print('[Error]: convert point cloud fail at:', model_path)

        return data