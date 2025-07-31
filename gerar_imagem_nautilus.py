#!/usr/bin/env python3
"""
🖼️ GERADOR DE IMAGEM NAUTILUS - Simples e rápido
===============================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def main():
    print("🖼️ GERANDO IMAGEM NAUTILUS...")
    
    try:
        # Carrega imagem original
        image = Image.open("figures/pikachu.png").convert('RGBA')
        
        # Carrega point cloud
        points = []
        with open('pikachu_pointcloud_nautilus.ply', 'r') as f:
            lines = f.readlines()
            reading_data = False
            for line in lines:
                if line.startswith('end_header'):
                    reading_data = True
                    continue
                if reading_data:
                    coords = line.strip().split()
                    if len(coords) >= 3:
                        points.append([float(coords[0]), float(coords[1]), float(coords[2])])
        
        points = np.array(points)
        
        # Cria visualização
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Imagem original
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(image)
        ax1.set_title('Pikachu 3D Original', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. Point cloud frontal
        ax2 = plt.subplot(2, 4, 2, projection='3d')
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='gold', s=2, alpha=0.8)
        ax2.set_title('☁️ Point Cloud (Frontal)', fontweight='bold')
        ax2.view_init(elev=0, azim=0)
        
        # 3. Point cloud lateral
        ax3 = plt.subplot(2, 4, 3, projection='3d')
        ax3.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='orange', s=2, alpha=0.8)
        ax3.set_title('☁️ Point Cloud (Lateral)', fontweight='bold')
        ax3.view_init(elev=0, azim=90)
        
        # 4. Point cloud superior
        ax4 = plt.subplot(2, 4, 4, projection='3d')
        ax4.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='red', s=2, alpha=0.8)
        ax4.set_title('☁️ Point Cloud (Superior)', fontweight='bold')
        ax4.view_init(elev=90, azim=0)
        
        # 5. Point cloud 3D
        ax5 = plt.subplot(2, 4, 5, projection='3d')
        ax5.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='purple', s=2, alpha=0.8)
        ax5.set_title('☁️ Point Cloud (3D)', fontweight='bold')
        ax5.view_init(elev=20, azim=45)
        
        # 6-8. Informações
        ax6 = plt.subplot(2, 4, 6)
        ax6.text(0.1, 0.8, '🔥 NAUTILUS REAL', fontsize=16, fontweight='bold')
        ax6.text(0.1, 0.6, f'☁️ {len(points):,} pontos', fontsize=12)
        ax6.text(0.1, 0.4, '🧠 Shape embeddings', fontsize=12)
        ax6.text(0.1, 0.2, '🔢 VQ quantização', fontsize=12)
        ax6.axis('off')
        
        ax7 = plt.subplot(2, 4, 7)
        ax7.text(0.1, 0.8, '📁 ARQUIVOS:', fontsize=14, fontweight='bold')
        ax7.text(0.1, 0.6, '• embeddings.npz', fontsize=10)
        ax7.text(0.1, 0.5, '• pointcloud.ply', fontsize=10)
        ax7.text(0.1, 0.4, '• mesh.obj', fontsize=10)
        ax7.text(0.1, 0.3, '• mesh.stl', fontsize=10)
        ax7.axis('off')
        
        ax8 = plt.subplot(2, 4, 8)
        ax8.text(0.1, 0.8, '✅ PIPELINE:', fontsize=14, fontweight='bold')
        ax8.text(0.1, 0.6, '1. Imagem → PC', fontsize=10)
        ax8.text(0.1, 0.5, '2. PC → Encoding', fontsize=10)
        ax8.text(0.1, 0.4, '3. Encoding → VQ', fontsize=10)
        ax8.text(0.1, 0.3, '4. VQ → Decoding', fontsize=10)
        ax8.text(0.1, 0.2, '5. Decoding → Mesh', fontsize=10)
        ax8.axis('off')
        
        plt.tight_layout()
        plt.suptitle('🔥 NAUTILUS PIPELINE - PIKACHU 3D PROCESSADO 🔥', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig('pikachu_nautilus_resultado.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Imagem gerada: pikachu_nautilus_resultado.png")
        
        # Estatísticas
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   ☁️ Point cloud: {len(points):,} pontos")
        print(f"   📐 Range X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"   📐 Range Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"   📐 Range Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        print("\n🎉 IMAGEM NAUTILUS GERADA COM SUCESSO!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
