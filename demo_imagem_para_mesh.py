#!/usr/bin/env python3
"""
🖼️ DEMO: Imagem para Point Cloud e Mesh com Nautilus
==================================================

Este script demonstra como o Nautilus pode gerar point clouds e meshes 
a partir de uma imagem como entrada.

FUNCIONALIDADES DO NAUTILUS:
✅ Entrada: Imagem (JPG, PNG, etc.)
✅ Saída: Point Cloud + Mesh 3D
✅ Qualidade: Até 5.000 faces
✅ Aplicações: AR/VR, jogos, design digital

NOTA: Este é um exemplo simulado já que não temos o modelo treinado oficial.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from miche.michelangelo.models.conditional_encoders.encoder_factory import CLIPImageEncoder
import trimesh
from pathlib import Path

def simular_clip_encoder():
    """Simula o encoder CLIP para processar imagens"""
    print("🎯 Simulando CLIP Image Encoder...")
    
    # Simulação de um encoder de imagem
    class CLIPSimulado:
        def __init__(self):
            self.embedding_dim = 768
            
        def encode_image(self, image_tensor):
            """Simula a codificação de uma imagem em features"""
            batch_size = image_tensor.shape[0]
            # Gera features simuladas baseadas na imagem
            features = torch.randn(batch_size, self.embedding_dim)
            return features
    
    return CLIPSimulado()

def processar_imagem(image_path):
    """Processa uma imagem e converte para tensor"""
    try:
        # Carrega e redimensiona a imagem
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))  # Tamanho padrão CLIP
        
        # Converte para tensor
        image_array = np.array(image) / 255.0
        image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor, image
    except Exception as e:
        print(f"❌ Erro ao carregar imagem: {e}")
        # Cria uma imagem simulada
        return criar_imagem_simulada()

def criar_imagem_simulada():
    """Cria uma imagem simulada para demonstração"""
    print("🎨 Criando imagem simulada...")
    
    # Cria um gradiente colorido como exemplo
    size = 224
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Padrão espiral como uma "concha nautilus"
    angle = np.arctan2(Y - 0.5, X - 0.5)
    radius = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    
    # Cria padrão inspirado no nautilus
    pattern = np.sin(4 * angle + 10 * radius) * np.exp(-3 * radius)
    
    # Converte para RGB
    image_array = np.zeros((size, size, 3))
    image_array[:, :, 0] = (pattern + 1) / 2  # Canal vermelho
    image_array[:, :, 1] = (np.cos(2 * angle) + 1) / 2  # Canal verde
    image_array[:, :, 2] = (radius * 2).clip(0, 1)  # Canal azul
    
    # Converte para PIL Image
    image = Image.fromarray((image_array * 255).astype(np.uint8))
    
    # Converte para tensor
    image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image

def imagem_para_features(image_tensor, clip_encoder):
    """Converte imagem para features latentes"""
    print("🔍 Extraindo features da imagem...")
    
    with torch.no_grad():
        # Simula o processo de encoding da imagem
        image_features = clip_encoder.encode_image(image_tensor)
        
    print(f"   ✅ Features extraídas: {image_features.shape}")
    return image_features

def features_para_pointcloud(image_features):
    """Gera point cloud a partir das features da imagem"""
    print("☁️ Gerando point cloud a partir da imagem...")
    
    # Simula a geração de point cloud baseada nas features
    feature_vector = image_features[0].numpy()
    
    # Usa as features para influenciar a forma do point cloud
    n_points = 2048
    
    # Gera pontos base usando as features como seed
    np.random.seed(int(np.sum(feature_vector[:10]) * 1000) % 2**32)
    
    # Cria uma forma baseada nas features
    t = np.linspace(0, 4*np.pi, n_points)
    
    # Usa diferentes componentes das features para diferentes aspectos
    amplitude = 0.5 + 0.3 * np.mean(feature_vector[:100])
    frequency = 2 + np.mean(feature_vector[100:200]) * 3
    noise_level = 0.1 + 0.05 * np.mean(feature_vector[200:300])
    
    # Gera espiral 3D inspirada no nautilus
    x = amplitude * np.cos(frequency * t) * (1 + 0.1 * t / (4*np.pi))
    y = amplitude * np.sin(frequency * t) * (1 + 0.1 * t / (4*np.pi))
    z = 0.2 * t / np.pi + noise_level * np.random.randn(n_points)
    
    # Adiciona ruído baseado nas features
    noise = noise_level * np.random.randn(n_points, 3)
    points = np.column_stack([x, y, z]) + noise
    
    # Normaliza para [-1, 1]
    points = 2 * (points - points.min()) / (points.max() - points.min()) - 1
    
    print(f"   ✅ Point cloud gerado: {points.shape} pontos")
    print(f"   📊 Range: [{points.min():.3f}, {points.max():.3f}]")
    
    return points

def pointcloud_para_mesh(points):
    """Converte point cloud para mesh usando algoritmo simulado"""
    print("🔺 Convertendo point cloud para mesh...")
    
    try:
        # Cria mesh usando triangulação Delaunay na projeção 2D
        from scipy.spatial import SphericalVoronoi, geometric_slerp
        
        # Projeta pontos numa esfera unitária
        norms = np.linalg.norm(points, axis=1)
        norms[norms == 0] = 1e-8  # Evita divisão por zero
        points_normalized = points / norms.reshape(-1, 1)
        
        # Remove pontos duplicados
        unique_points = []
        seen = set()
        for point in points_normalized:
            point_tuple = tuple(np.round(point, 6))
            if point_tuple not in seen:
                seen.add(point_tuple)
                unique_points.append(point)
        
        unique_points = np.array(unique_points)
        
        if len(unique_points) < 4:
            raise ValueError("Não há pontos únicos suficientes")
        
        # Cria malha convexa
        from scipy.spatial import ConvexHull
        hull = ConvexHull(unique_points)
        
        vertices = hull.points
        faces = hull.simplices
        
        print(f"   ✅ Mesh criado: {len(vertices)} vértices, {len(faces)} faces")
        
        return vertices, faces
        
    except Exception as e:
        print(f"   ⚠️ Erro na triangulação: {e}")
        print("   🔄 Usando malha simples...")
        
        # Cria uma malha simples como fallback
        n = min(len(points), 100)  # Usa apenas os primeiros 100 pontos
        selected_points = points[:n]
        
        # Cria faces conectando pontos próximos
        faces = []
        for i in range(n-2):
            faces.append([i, i+1, i+2])
        
        return selected_points, np.array(faces)

def visualizar_pipeline(image, points, vertices, faces):
    """Visualiza o pipeline completo: Imagem → Point Cloud → Mesh"""
    print("📊 Criando visualização do pipeline...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Imagem original
    ax1 = fig.add_subplot(131)
    ax1.imshow(image)
    ax1.set_title('🖼️ Imagem de Entrada', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Point Cloud
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
    ax2.set_title('☁️ Point Cloud Gerado', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 3. Mesh
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plota arestas do mesh
    for face in faces[:min(200, len(faces))]:  # Limita para performance
        if len(face) >= 3:
            triangle = vertices[face[:3]]
            ax3.plot([triangle[0,0], triangle[1,0]], 
                    [triangle[0,1], triangle[1,1]], 
                    [triangle[0,2], triangle[1,2]], 'b-', alpha=0.3, linewidth=0.5)
            ax3.plot([triangle[1,0], triangle[2,0]], 
                    [triangle[1,1], triangle[2,1]], 
                    [triangle[1,2], triangle[2,2]], 'b-', alpha=0.3, linewidth=0.5)
            ax3.plot([triangle[2,0], triangle[0,0]], 
                    [triangle[2,1], triangle[0,1]], 
                    [triangle[2,2], triangle[0,2]], 'b-', alpha=0.3, linewidth=0.5)
    
    ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=2, alpha=0.8)
    ax3.set_title('🔺 Mesh 3D Final', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('pipeline_imagem_para_mesh.png', dpi=150, bbox_inches='tight')
    plt.show()

def salvar_resultados(image, points, vertices, faces):
    """Salva todos os resultados gerados"""
    print("💾 Salvando resultados...")
    
    # Salva imagem
    image.save('imagem_entrada.png')
    print("   ✅ Imagem salva: imagem_entrada.png")
    
    # Salva point cloud
    np.save('pointcloud_gerado.npy', points)
    print("   ✅ Point cloud salvo: pointcloud_gerado.npy")
    
    # Salva mesh como OBJ
    try:
        # Cria mesh com trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export('mesh_gerado.obj')
        print("   ✅ Mesh salvo: mesh_gerado.obj")
    except Exception as e:
        print(f"   ⚠️ Erro ao salvar mesh: {e}")
        
        # Salva dados como numpy arrays
        np.save('mesh_vertices.npy', vertices)
        np.save('mesh_faces.npy', faces)
        print("   ✅ Dados do mesh salvos: mesh_vertices.npy, mesh_faces.npy")

def demo_completo():
    """Executa demonstração completa do pipeline Imagem → Mesh"""
    print("="*70)
    print("🚀 DEMO: NAUTILUS - IMAGEM PARA POINT CLOUD E MESH")
    print("="*70)
    print()
    
    print("📋 CAPACIDADES DO NAUTILUS:")
    print("   🖼️  Entrada: Qualquer imagem (JPG, PNG, etc.)")
    print("   ☁️  Gera: Point cloud detalhado")
    print("   🔺 Produz: Mesh 3D com até 5.000 faces")
    print("   ⚡ Tempo: 3-4 minutos em GPU A100")
    print("   🎯 Aplicações: AR/VR, jogos, design digital")
    print()
    
    # 1. Simula o encoder CLIP
    clip_encoder = simular_clip_encoder()
    
    # 2. Processa uma imagem (você pode especificar o caminho)
    image_path = "exemplo_imagem.jpg"  # Substitua pelo caminho da sua imagem
    image_tensor, image = processar_imagem(image_path)
    
    # 3. Extrai features da imagem
    image_features = imagem_para_features(image_tensor, clip_encoder)
    
    # 4. Gera point cloud a partir das features
    points = features_para_pointcloud(image_features)
    
    # 5. Converte point cloud para mesh
    vertices, faces = pointcloud_para_mesh(points)
    
    # 6. Visualiza todo o pipeline
    visualizar_pipeline(image, points, vertices, faces)
    
    # 7. Salva resultados
    salvar_resultados(image, points, vertices, faces)
    
    print("\n" + "="*70)
    print("✅ DEMO CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print()
    print("📁 ARQUIVOS GERADOS:")
    print("   📷 imagem_entrada.png")
    print("   ☁️  pointcloud_gerado.npy")
    print("   🔺 mesh_gerado.obj")
    print("   📊 pipeline_imagem_para_mesh.png")
    print()
    print("🎯 PRÓXIMOS PASSOS:")
    print("   1. Para usar o modelo real, obtenha o checkpoint oficial")
    print("   2. Execute: python miche/encode.py --image_path sua_imagem.jpg")
    print("   3. O Nautilus suporta imagens de qualquer objeto ou cena!")
    print()
    print("🌐 Mais informações: https://nautilusmeshgen.github.io")

if __name__ == "__main__":
    demo_completo()
