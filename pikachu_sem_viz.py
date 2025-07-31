#!/usr/bin/env python3
"""
Pikachu corrigido - SEM visualização (só gera arquivos)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import trimesh

def processar_pikachu_sem_visualizacao():
    """Processa o Pikachu sem mostrar gráficos"""
    print("🔧 PROCESSANDO PIKACHU (sem visualização)...")
    
    try:
        # Carrega imagem
        print("📷 Carregando imagem...")
        image = Image.open("figures/pikachu.png").convert('RGBA')
        img_array = np.array(image)
        
        print(f"   ✅ Imagem: {image.size}")
        
        # Cria máscara
        if img_array.shape[2] == 4:
            alpha = img_array[:, :, 3]
            mask = alpha > 128
        else:
            img_rgb = img_array[:, :, :3]
            white_mask = np.all(img_rgb > 240, axis=2)
            mask = ~white_mask
        
        print(f"   🎯 Pixels válidos: {np.sum(mask)}")
        
        # Gera point cloud
        print("☁️ Gerando point cloud...")
        h, w = mask.shape
        points = []
        colors = []
        
        step = 5  # Densidade maior
        for y in range(0, h, step):
            for x in range(0, w, step):
                if mask[y, x]:
                    x_norm = (x / w) * 2 - 1
                    y_norm = -(y / h) * 2 + 1
                    
                    # Profundidade baseada na distância do centro
                    center_x, center_y = 0, 0
                    dist = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2)
                    z_depth = 0.3 * (1 - dist) * np.random.uniform(0.8, 1.2)
                    
                    # Múltiplas camadas Z
                    for z in np.linspace(-z_depth, z_depth, 3):
                        points.append([x_norm, y_norm, z])
                        colors.append([1.0, 1.0, 0.0])  # Amarelo
        
        points = np.array(points)
        colors = np.array(colors)
        
        print(f"   ✅ Point cloud: {len(points)} pontos")
        
        # Gera mesh simples usando ConvexHull
        print("🔺 Gerando mesh...")
        if len(points) >= 4:
            try:
                hull = ConvexHull(points)
                vertices = hull.points
                faces = hull.simplices
                
                print(f"   ✅ Mesh: {len(vertices)} vértices, {len(faces)} faces")
                
                # Salva arquivos
                print("💾 Salvando arquivos...")
                
                # Point cloud
                np.save('pikachu_pointcloud_corrigido.npy', points)
                print("   ✅ pikachu_pointcloud_corrigido.npy")
                
                # Mesh
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh.export('pikachu_mesh_corrigido.obj')
                mesh.export('pikachu_mesh_corrigido.stl')
                
                print("   ✅ pikachu_mesh_corrigido.obj")
                print("   ✅ pikachu_mesh_corrigido.stl")
                
                # Visualização salva (sem mostrar)
                fig = plt.figure(figsize=(15, 5))
                
                # Imagem original
                ax1 = plt.subplot(1, 3, 1)
                ax1.imshow(image)
                ax1.set_title('Pikachu Original')
                ax1.axis('off')
                
                # Point cloud
                ax2 = plt.subplot(1, 3, 2, projection='3d')
                ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c='yellow', s=1, alpha=0.8)
                ax2.set_title('Point Cloud')
                ax2.view_init(elev=20, azim=45)
                
                # Mesh wireframe
                ax3 = plt.subplot(1, 3, 3, projection='3d')
                
                # Plota algumas faces
                for i, face in enumerate(faces[:min(100, len(faces))]):
                    if all(f < len(vertices) for f in face):
                        triangle = vertices[face]
                        ax3.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                                       alpha=0.6, color='gold')
                
                ax3.set_title('Mesh 3D')
                ax3.view_init(elev=20, azim=45)
                
                plt.tight_layout()
                plt.savefig('pikachu_corrigido.png', dpi=200, bbox_inches='tight')
                plt.close()  # Fecha sem mostrar
                
                print("   ✅ pikachu_corrigido.png")
                
                print("\n🎉 PIKACHU CORRIGIDO GERADO COM SUCESSO!")
                print("\n📁 ARQUIVOS CRIADOS:")
                print("   🖼️ pikachu_corrigido.png")
                print("   ☁️ pikachu_pointcloud_corrigido.npy") 
                print("   🔺 pikachu_mesh_corrigido.obj")
                print("   🖨️ pikachu_mesh_corrigido.stl")
                
            except Exception as e:
                print(f"   ❌ Erro no mesh: {e}")
                
        else:
            print("   ❌ Pontos insuficientes para mesh")
            
    except Exception as e:
        print(f"❌ ERRO GERAL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    processar_pikachu_sem_visualizacao()
