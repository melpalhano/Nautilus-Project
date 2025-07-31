#!/usr/bin/env python3
"""
🔴 POINT CLOUD SUPERIOR - Vista de cima do Pikachu Nautilus
==========================================================

Mostra apenas o point cloud na vista superior (mais bonita!)
"""

import numpy as np
import matplotlib
# Para visualização INTERATIVA, remove o 'Agg'
# matplotlib.use('Agg')  # Comentado para permitir interação
import matplotlib.pyplot as plt
from PIL import Image

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
        print(f"   ✅ {len(points):,} pontos carregados")
        return points
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def gerar_visualizacao_superior(points):
    """Gera visualização focada na vista superior"""
    print("🎨 GERANDO VISUALIZAÇÃO SUPERIOR...")
    
    # Cria figura grande para destaque
    fig = plt.figure(figsize=(16, 12))
    
    # Carrega imagem original para referência
    try:
        image = Image.open("figures/pikachu.png").convert('RGBA')
        
        # Layout: Imagem original + Point cloud superior grande
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(image)
        ax1.set_title('Pikachu 3D Original', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
    except:
        print("   ⚠️ Imagem original não encontrada")
    
    # Point cloud superior - DESTAQUE PRINCIPAL
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    
    # Plot com cores vibrantes
    scatter = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=points[:, 2], # Cor baseada na altura Z
                         cmap='Reds',    # Vermelho como na imagem
                         s=8,            # Pontos maiores
                         alpha=0.9,      # Mais opaco
                         edgecolors='darkred', # Borda escura
                         linewidth=0.1)
    
    # Vista superior (elev=90 = olhando de cima)
    ax2.view_init(elev=90, azim=0)
    
    # Configurações visuais
    ax2.set_title('🔴 POINT CLOUD SUPERIOR\n(Vista de Cima - Nautilus)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Remove eixos para visual mais limpo
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([0, 1])
    
    # Grid sutil
    ax2.grid(True, alpha=0.3)
    
    # Colorbar para mostrar profundidade
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8, pad=0.1)
    cbar.set_label('Profundidade Z', fontsize=12, fontweight='bold')
    

    
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
    plt.savefig('pikachu_pointcloud_superior.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Visualização salva: pikachu_pointcloud_superior.png")

def criar_versao_interativa(points):
    """Cria versão com múltiplos ângulos da vista superior"""
    print("🎨 CRIANDO VERSÃO COM MÚLTIPLOS ÂNGULOS...")
    
    fig = plt.figure(figsize=(20, 10))
    
    # 5 vistas superiores com rotações diferentes
    angulos = [0, 45, 90, 135, 180]
    cores = ['Reds', 'Oranges', 'YlOrRd', 'OrRd', 'Reds']
    
    for i, (angulo, cor) in enumerate(zip(angulos, cores)):
        ax = plt.subplot(1, 5, i+1, projection='3d')
        
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=points[:, 2], 
                           cmap=cor,
                           s=4, 
                           alpha=0.8)
        
        # Vista superior com rotação
        ax.view_init(elev=90, azim=angulo)
        
        ax.set_title(f'Vista Superior\n(Rotação {angulo}°)', 
                    fontsize=12, fontweight='bold')
        
        # Remove eixos para visual limpo
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1])
    
    fig.suptitle('🔴 PIKACHU POINT CLOUD - VISTAS SUPERIORES 🔴', 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pikachu_pointcloud_superior_multiplo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Múltiplas vistas salvas: pikachu_pointcloud_superior_multiplo.png")

def visualizacao_interativa_3d(points):
    """Cria visualização 3D TOTALMENTE INTERATIVA"""
    print("🎮 INICIANDO VISUALIZAÇÃO INTERATIVA 3D...")
    
    # Ativa modo interativo
    plt.ion()
    
    # Cria figura grande
    fig = plt.figure(figsize=(14, 10))
  
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Point cloud com cores vibrantes
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=points[:, 2],     # Cor baseada na altura
                        cmap='plasma',      # Colormap bonito
                        s=12,               # Pontos maiores
                        alpha=0.8,          # Transparência
                        edgecolors='black', # Borda
                        linewidth=0.1)
    
    # Configurações dos eixos
    ax.set_xlabel('🔴 Eixo X', fontsize=12, fontweight='bold')
    ax.set_ylabel('🔵 Eixo Y', fontsize=12, fontweight='bold') 
    ax.set_zlabel('🟢 Eixo Z', fontsize=12, fontweight='bold')
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Altura (Z)', fontsize=12, fontweight='bold')
    
    # Estatísticas
    fig.text(0.02, 0.02, f"""
NAUTILUS STATS:
• {len(points):,} pontos
• Range X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]
• Range Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]
• Range Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]
    """, fontsize=10, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
    
    # Vista inicial interessante
    ax.view_init(elev=30, azim=45)
    
    print("\n" + "="*60)
    print("� VISUALIZAÇÃO INTERATIVA ATIVA!")
    print("="*60)
    print("🖱️  Use o MOUSE para ROTACIONAR o point cloud")
    print("� Use o SCROLL para ZOOM in/out")
    print("📐 SHIFT + MOUSE para MOVER/PAN")
    print("❌ FECHE a janela quando terminar")
    print("="*60)
    print("🔥 INTERAJA COM O PIKACHU NAUTILUS! 🔥")
    
    # Mantém janela aberta para interação
    plt.show(block=True)
    
    print("✅ Visualização interativa finalizada!")

def main():
    """Função principal"""
    print("🔴 POINT CLOUD SUPERIOR - INICIANDO...")
    print("="*60)
    
    # Carrega point cloud
    points = carregar_pointcloud()
    
    if points is None:
        print("❌ Falha ao carregar point cloud!")
        return
    
    # Menu de escolha
    print("\n🎯 ESCOLHA SUA VISUALIZAÇÃO:")
    print("1️⃣ Salvar imagens estáticas (atual)")
    print("2️⃣ Visualização 3D INTERATIVA")
    print("3️⃣ Ambas as opções")
    
    while True:
        try:
            escolha = input("\n👉 Digite sua escolha (1-3): ").strip()
            if escolha in ['1', '2', '3']:
                break
            else:
                print("❌ Digite 1, 2 ou 3!")
        except:
            print("❌ Digite um número válido!")
    
    if escolha in ['1', '3']:
        # Gera visualização estática
        gerar_visualizacao_superior(points)
        criar_versao_interativa(points)
        
        print("\n" + "="*60)
        print("🎉 IMAGENS ESTÁTICAS GERADAS!")
        print("="*60)
        print("� ARQUIVOS CRIADOS:")
        print("   🔴 pikachu_pointcloud_superior.png")
        print("   � pikachu_pointcloud_superior_multiplo.png")
    
    if escolha in ['2', '3']:
        # Inicia visualização interativa
        print("\n🎮 INICIANDO MODO INTERATIVO...")
        visualizacao_interativa_3d(points)
    
    print("\n🔥 POINT CLOUD NAUTILUS COMPLETO! 🔥")

if __name__ == "__main__":
    main()
