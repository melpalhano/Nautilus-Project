#!/usr/bin/env python3
"""
🎮 VISUALIZADOR INTERATIVO 3D - Pikachu Point Cloud
===================================================

Abre visualização interativa onde você pode:
- Rotacionar com mouse
- Zoom com scroll
- Mover com teclas
- Vista em tempo real!
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

def carregar_pointcloud():
    """Carrega o point cloud do Nautilus"""
    print("☁️ CARREGANDO POINT CLOUD PARA VISUALIZAÇÃO INTERATIVA...")
    
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
        print(f"   ✅ {len(points):,} pontos carregados para interação!")
        return points
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def visualizacao_interativa(points):
    """Cria visualização 3D interativa"""
    print("🎮 INICIANDO VISUALIZADOR INTERATIVO 3D...")
    
    # Configura matplotlib para modo interativo
    plt.ion()
    
    # Cria figura grande para melhor interação
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('🎮 PIKACHU POINT CLOUD - VISUALIZAÇÃO INTERATIVA 3D 🎮', 
                fontsize=16, fontweight='bold')
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Point cloud com cores baseadas na altura
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=points[:, 2],     # Cor baseada na altura Z
                        cmap='coolwarm',    # Colormap bonito
                        s=15,               # Pontos maiores para melhor visualização
                        alpha=0.8,          # Transparência
                        edgecolors='black', # Borda preta
                        linewidth=0.1)
    
    # Configurações dos eixos
    ax.set_xlabel('🔴 Eixo X', fontsize=12, fontweight='bold')
    ax.set_ylabel('🔵 Eixo Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('🟢 Eixo Z', fontsize=12, fontweight='bold')
    
    # Limites dos eixos
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    
    # Grid para melhor orientação
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('🌈 Altura (Z)', fontsize=12, fontweight='bold')
    
    # Texto de instruções
    instructions = """
🎮 CONTROLES INTERATIVOS:
🖱️  Mouse: Arrastar para rotacionar
🔄 Scroll: Zoom in/out
📐 Botão direito: Pan/mover
⌨️  Teclas: Use toolbar para mais opções

🔥 NAUTILUS POINT CLOUD INTERATIVO!
"""
    
    fig.text(0.02, 0.02, instructions, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
    
    # Estatísticas do point cloud
    stats_text = f"""
📊 ESTATÍSTICAS:
   • Pontos: {len(points):,}
   • X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]
   • Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]
   • Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]
"""
    
    fig.text(0.02, 0.85, stats_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    # Vista inicial interessante
    ax.view_init(elev=30, azim=45)
    
    print("\n" + "="*60)
    print("🎮 VISUALIZADOR INTERATIVO ATIVO!")
    print("="*60)
    print("🖱️  Use o MOUSE para rotacionar o point cloud")
    print("🔄 Use o SCROLL para zoom in/out")
    print("📐 Botão DIREITO para mover/pan")
    print("❌ FECHE a janela para terminar")
    print("="*60)
    
    # Mantém a janela aberta para interação
    plt.show(block=True)
    
    print("✅ Visualização interativa finalizada!")

def main():
    """Função principal"""
    print("🎮 VISUALIZADOR INTERATIVO 3D - INICIANDO...")
    print("="*60)
    
    # Carrega point cloud
    points = carregar_pointcloud()
    
    if points is None:
        print("❌ Falha ao carregar point cloud!")
        return
    
    # Inicia visualização interativa
    visualizacao_interativa(points)
    
    print("\n🎉 VISUALIZAÇÃO INTERATIVA CONCLUÍDA!")

if __name__ == "__main__":
    main()
