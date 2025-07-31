#!/usr/bin/env python3
"""
Versão modificada do infer_pc.py para funcionar com modelo simulado
"""

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
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--mesh_path', type=str, default=None)
parser.add_argument('--pc_path', type=str, required=True)
parser.add_argument('--output_path', type=str, default='outputs')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--temperature', type=float, default=1)
args = parser.parse_args()

def main():
    """Função principal de inferência simulada"""
    print("🚀 NAUTILUS - INFERÊNCIA SIMULADA")
    print("=" * 40)
    
    # Criar diretório de output
    os.makedirs(args.output_path, exist_ok=True)
    
    try:
        # 1. Carregar configuração
        print("📋 Carregando configuração...")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✅ Config carregada: {args.config}")
        
        # 2. Carregar modelo
        print("🧠 Carregando modelo...")
        checkpoint = torch.load(args.model_path, map_location='cpu')
        
        # Verificar se é modelo dummy
        if 'info' in checkpoint and checkpoint['info'].get('type') == 'dummy_model':
            print("   ⚠️ ATENÇÃO: Modelo simulado detectado!")
            print("   ⚠️ Resultados serão aleatórios, não reais")
        
        # Criar modelo
        model = MeshTransformer(
            dim=512,  # Configuração do modelo dummy
            max_seq_len=config['max_seq_len'],
            attn_depth=8,
            u_size=config['u_size'],
            v_size=config['v_size']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Modelo carregado: {total_params:,} parâmetros")
        
        # 3. Carregar nuvem de pontos
        print("☁️ Carregando nuvem de pontos...")
        points = np.load(args.pc_path)
        print(f"   ✅ Pontos carregados: {points.shape}")
        print(f"   📏 Range: [{points.min():.3f}, {points.max():.3f}]")
        
        # 4. Simulação de processamento (já que não temos encoder real)
        print("🔄 Processando entrada...")
        
        # Simular encoding da nuvem de pontos
        # Em um modelo real, isso seria feito pelo encoder Michelangelo
        batch_size = args.batch_size
        seq_len = 1000  # Sequência simulada
        
        print(f"   🔄 Simulando encoding de nuvem de pontos...")
        fake_tokens = torch.randint(0, config['u_size'], (batch_size, seq_len))
        print(f"   ✅ Tokens simulados: {fake_tokens.shape}")
        
        # 5. Simulação de geração
        print("🎨 Gerando mesh...")
        with torch.no_grad():
            # Simular processo de geração
            embeddings = model.token_embed(fake_tokens)
            print(f"   ✅ Embeddings gerados: {embeddings.shape}")
            
            # Simular tokens de saída (em um modelo real, viria do transformer)
            output_tokens = torch.randint(0, config['u_size'] + config['v_size'], (batch_size, seq_len))
            print(f"   ✅ Tokens de output: {output_tokens.shape}")
        
        # 6. Detokenização simulada
        print("🔧 Detokenizando...")
        try:
            # Converter para numpy
            output_sequence = output_tokens[0].numpy()
            
            # Usar função de detokenização do projeto
            vertices = detokenize(
                output_sequence, 
                u_size=config['u_size'], 
                v_size=config['v_size']
            )
            
            print(f"   ✅ Vértices gerados: {vertices.shape}")
            
            # 7. Criar mesh simulada
            if len(vertices) >= 3:
                # Criar faces simples (triangulação básica)
                n_vertices = len(vertices)
                faces = []
                
                # Criar faces triangulares simples
                for i in range(0, n_vertices - 2, 3):
                    if i + 2 < n_vertices:
                        faces.append([i, i + 1, i + 2])
                
                faces = np.array(faces)
                
                if len(faces) > 0:
                    # Criar mesh com trimesh
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    
                    # Salvar mesh
                    output_file = os.path.join(args.output_path, 'output_mesh_simulado.ply')
                    mesh.export(output_file)
                    
                    print(f"   ✅ Mesh salva: {output_file}")
                    print(f"   📊 Vértices: {len(vertices)}, Faces: {len(faces)}")
                    
                    # Salvar também como numpy para inspeção
                    np.save(os.path.join(args.output_path, 'vertices_simulado.npy'), vertices)
                    np.save(os.path.join(args.output_path, 'faces_simulado.npy'), faces)
                    
                else:
                    print("   ⚠️ Não foi possível criar faces válidas")
            else:
                print("   ⚠️ Não há vértices suficientes para criar mesh")
            
        except Exception as e:
            print(f"   ❌ Erro na detokenização: {e}")
            print("   🔄 Criando mesh alternativa...")
            
            # Fallback: criar mesh simples baseada na nuvem de pontos original
            # Simplificar pontos para criar uma mesh básica
            subsample_idx = np.random.choice(len(points), min(len(points), 100), replace=False)
            vertices = points[subsample_idx]
            
            # Criar faces simples
            faces = []
            for i in range(0, len(vertices) - 2, 3):
                faces.append([i, i + 1, i + 2])
            
            faces = np.array(faces)
            
            if len(faces) > 0:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                output_file = os.path.join(args.output_path, 'output_mesh_fallback.ply')
                mesh.export(output_file)
                print(f"   ✅ Mesh fallback salva: {output_file}")
        
        print("\n" + "=" * 40)
        print("🎉 INFERÊNCIA SIMULADA COMPLETADA!")
        print(f"📁 Resultados salvos em: {args.output_path}")
        print("\n⚠️ LEMBRE-SE:")
        print("   • Este é um resultado SIMULADO")
        print("   • Mesh gerada é aleatória/baseada em fallback")
        print("   • Para resultados reais, precisa do modelo oficial")
        print("=" * 40)
        
    except Exception as e:
        print(f"❌ Erro durante inferência: {e}")
        print("\n🔧 DIAGNÓSTICO:")
        print(f"   • Config: {args.config}")
        print(f"   • Modelo: {args.model_path}")
        print(f"   • PC: {args.pc_path}")
        print("   • Verifique se todos os arquivos existem")

if __name__ == "__main__":
    main()
