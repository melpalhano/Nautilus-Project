#!/usr/bin/env python3
"""
🔥 GERADOR + INTERATIVO - Pikachu Nautilus
=========================================

1. GERA a imagem do Point Cloud Superior
2. ABRE visualização INTERATIVA 3D
3. Tudo em um só script!
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import threading
import time

def carregar_pointcloud():
    """Carrega o point cloud do Nautilus"""
    print("☁️ CARREGANDO POINT CLOUD NAUTILUS...")
    
    points = []
    try:
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
                        x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                        points.append([x, y, z])
        
        points = np.array(points)
        print(f"   ✅ {len(points):,} pontos carregados!")
        return points
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def gerar_imagem_estatica(points):
    """Gera a imagem estática do Point Cloud Superior"""
    print("🎨 GERANDO IMAGEM ESTÁTICA...")
    
    # Usa backend não-interativo para salvar
    matplotlib.use('Agg')
    
    # Cria figura grande
    fig = plt.figure(figsize=(16, 12))
    
    # Carrega imagem original
    try:
        image = Image.open("figures/pikachu.png").convert('RGBA')
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(image)
        ax1.set_title('🎯 Pikachu 3D Original', fontsize=16, fontweight='bold')
        ax1.axis('off')
    except:
        print("   ⚠️ Imagem original não encontrada")
    
    # Point cloud superior
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    
    scatter = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=points[:, 2], 
                         cmap='Reds',
                         s=8,
                         alpha=0.9,
                         edgecolors='darkred',
                         linewidth=0.1)
    
    # Vista superior
    ax2.view_init(elev=90, azim=0)
    
    ax2.set_title('🔴 POINT CLOUD SUPERIOR\n(Vista de Cima - Nautilus)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1]) 
    ax2.set_zlim([0, 1])
    
    ax2.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8, pad=0.1)
    cbar.set_label('Profundidade Z', fontsize=12, fontweight='bold')
    
    # Título geral
    fig.suptitle('PIKACHU NAUTILUS - POINT CLOUD ', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # Informações técnicas
    info_text = f"""
📊 ESTATÍSTICAS:
   • Pontos: {len(points):,}
   • Range X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]
   • Range Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]
   • Range Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]
   
🔥 PROCESSADO COM NAUTILUS REAL
   • Pipeline: Imagem → Point Cloud → Embeddings
   • Formato: PLY (padrão indústria)
   • Vista: Superior (90° elevação)
"""
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('pikachu_superior_completo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Imagem salva: pikachu_superior_completo.png")

def visualizacao_interativa_3d(points):
    """Cria visualização 3D TOTALMENTE INTERATIVA"""
    print("🎮 ABRINDO VISUALIZAÇÃO INTERATIVA 3D...")
    
    # Muda para backend interativo
    matplotlib.use('TkAgg')
    plt.ion()
    
    # Cria figura interativa
    fig = plt.figure(figsize=(14, 10))

    
    ax = fig.add_subplot(111, projection='3d')
    
    # Point cloud com cores vibrantes
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=points[:, 2],     
                        cmap='plasma',      
                        s=15,               
                        alpha=0.8,          
                        edgecolors='black', 
                        linewidth=0.1)
    
    # Configurações dos eixos
    ax.set_xlabel('Eixo X', fontsize=14, fontweight='bold')
    ax.set_ylabel('Eixo Y', fontsize=14, fontweight='bold') 
    ax.set_zlabel('Eixo Z', fontsize=14, fontweight='bold')
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    
    # Grid elegante
    ax.grid(True, alpha=0.4)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Profundidagem (Z)', fontsize=14, fontweight='bold')
    


    
    # Estatísticas em destaque
    fig.text(0.02, 0.02, f"""
NAUTILUS REAL - ESTATÍSTICAS:
• Total de pontos: {len(points):,}
• Range X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]
• Range Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]
• Range Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]

    """, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9))
    
    # Vista inicial espetacular
    ax.view_init(elev=25, azim=45)
    
    print("\n" + "🔥"*50)
    print("🎮 VISUALIZAÇÃO INTERATIVA ATIVA!")
    print("🔥"*50)
    print("🖱️  ARRASTAR = Rotacionar o Pikachu!")
    print("🔄 SCROLL = Zoom suave")
    print("📐 SHIFT+MOUSE = Mover visualização")
    print("❌ FECHE a janela quando terminar")
    print("🔥"*50)
    print("🚀 INTERAJA COM O POINT CLOUD NAUTILUS!")
    print("🔥"*50)
    
    # Mantém janela aberta
    plt.show(block=True)
    
    print("✅ Visualização interativa finalizada!")

def executar_ambos(points):
    """Executa geração de imagem E visualização interativa"""
    print("🚀 EXECUTANDO: GERAÇÃO + INTERAÇÃO")
    print("="*60)
    
    # 1. Gera imagem estática primeiro
    gerar_imagem_estatica(points)
    
    # 2. Pequena pausa
    print("\n⏳ Preparando visualização interativa...")
    time.sleep(1)
    
    # 3. Abre visualização interativa
    visualizacao_interativa_3d(points)

def main():
    """Função principal"""
    print("🔥 GERADOR + INTERATIVO - PIKACHU NAUTILUS")
    print("="*60)
    print("📋 Este script irá:")
    print("   1️⃣ GERAR imagem estática do Point Cloud Superior")
    print("   2️⃣ ABRIR visualização 3D INTERATIVA")
    print("   3️⃣ Permitir rotação, zoom e exploração!")
    print("="*60)
    
    # Carrega point cloud
    points = carregar_pointcloud()
    
    if points is None:
        print("❌ Falha ao carregar point cloud!")
        return
    
    # Executa automaticamente ambos
    print("🚀 EXECUTANDO AUTOMATICAMENTE...")
    executar_ambos(points)
    
    print("\n" + "🎉"*20)
    print("🎉 PROCESSO COMPLETO FINALIZADO!")
    print("🎉"*20)
    print("📸 Imagem salva: pikachu_superior_completo.png")
    print("🎮 Visualização interativa executada!")
    print("🔥 NAUTILUS POINT CLOUD MASTER! 🔥")

if __name__ == "__main__":
    main()
