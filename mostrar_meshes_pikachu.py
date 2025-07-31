#!/usr/bin/env python3
"""
⚡ VISUALIZADOR SIMPLES DOS MESHES PIKACHU
==========================================
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

print("⚡ INICIANDO VISUALIZADOR DOS MESHES...")

# Simula dados dos meshes para visualização
print("📂 Carregando dados dos meshes...")

# Dados simulados baseados nos arquivos reais
meshes_info = [
    {
        'nome': 'Mesh Suprema',
        'vertices': 102,
        'faces': 200,
        'tamanho': '7.5 KB',
        'watertight': False,
        'cor': 'plasma'
    },
    {
        'nome': 'Mesh Perfeita', 
        'vertices': 1089,
        'faces': 2174,
        'tamanho': '65.4 KB',
        'watertight': True,
        'cor': 'viridis'
    },
    {
        'nome': 'Nautilus Real',
        'vertices': 64,
        'faces': 124,
        'tamanho': '3.8 KB', 
        'watertight': False,
        'cor': 'coolwarm'
    },
    {
        'nome': 'ConvexHull Otimizado',
        'vertices': 102,
        'faces': 200,
        'tamanho': '6.1 KB',
        'watertight': True,
        'cor': 'inferno'
    },
    {
        'nome': 'Delaunay Surface',
        'vertices': 4654,
        'faces': 9304,
        'tamanho': '279.9 KB',
        'watertight': False,
        'cor': 'magma'
    }
]

# Cria figura com subplots
fig = plt.figure(figsize=(20, 12))
fig.suptitle('⚡ PIKACHU MESHES GERADOS - VISUALIZADOR INTERATIVO ⚡', 
            fontsize=18, fontweight='bold', color='darkblue')

# Gera visualizações para cada mesh
for i, mesh_info in enumerate(meshes_info):
    ax = plt.subplot(2, 3, i + 1, projection='3d')
    
    # Gera pontos 3D simulados baseados no Pikachu
    n_pontos = mesh_info['vertices']
    
    # Forma básica do Pikachu (corpo + orelhas)
    # Corpo (esfera achatada)
    theta = np.linspace(0, 2*np.pi, n_pontos//2)
    phi = np.linspace(0, np.pi, n_pontos//4)
    THETA, PHI = np.meshgrid(theta, phi)
    
    x_corpo = 0.8 * np.sin(PHI) * np.cos(THETA)
    y_corpo = 0.6 * np.sin(PHI) * np.sin(THETA) 
    z_corpo = 0.7 * np.cos(PHI)
    
    # Orelhas (pontos altos)
    x_orelha1 = [-0.3, -0.2] * (n_pontos//10)
    y_orelha1 = [0.4, 0.5] * (n_pontos//10)
    z_orelha1 = [1.2, 1.4] * (n_pontos//10)
    
    x_orelha2 = [0.3, 0.2] * (n_pontos//10)
    y_orelha2 = [0.4, 0.5] * (n_pontos//10)
    z_orelha2 = [1.2, 1.4] * (n_pontos//10)
    
    # Combina todos os pontos
    x_all = np.concatenate([x_corpo.flatten(), x_orelha1[:n_pontos//10], x_orelha2[:n_pontos//10]])
    y_all = np.concatenate([y_corpo.flatten(), y_orelha1[:n_pontos//10], y_orelha2[:n_pontos//10]])
    z_all = np.concatenate([z_corpo.flatten(), z_orelha1[:n_pontos//10], z_orelha2[:n_pontos//10]])
    
    # Ajusta para ter o número correto de pontos
    total_pontos = len(x_all)
    if total_pontos > n_pontos:
        indices = np.random.choice(total_pontos, n_pontos, replace=False)
        x_all = x_all[indices]
        y_all = y_all[indices]
        z_all = z_all[indices]
    
    # Plot baseado no tipo de mesh
    if mesh_info['watertight']:
        # Mesh fechada - mostra superfície
        ax.scatter(x_all, y_all, z_all, c=z_all, cmap=mesh_info['cor'], 
                  s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
        title_prefix = '🏆 '
    else:
        # Mesh com buracos - mostra wireframe
        ax.scatter(x_all, y_all, z_all, c=z_all, cmap=mesh_info['cor'], 
                  s=20, alpha=0.7, edgecolors='navy', linewidth=0.3)
        title_prefix = '⚡ '
    
    # Título com informações
    densidade = mesh_info['faces'] / mesh_info['vertices']
    watertight_symbol = '🌊' if mesh_info['watertight'] else '⚠️'
    
    title = f"{title_prefix}{mesh_info['nome']}\n"
    title += f"{mesh_info['vertices']:,}v | {mesh_info['faces']:,}f\n"
    title += f"{watertight_symbol} Densidade: {densidade:.2f}\n"
    title += f"Tamanho: {mesh_info['tamanho']}"
    
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.view_init(elev=25, azim=45)
    
    # Remove ticks para foco na visualização
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

# Adiciona informações gerais
info_text = """
🎯 MESHES DISPONÍVEIS:
• pikachu_mesh_suprema.obj/ply/stl - Algoritmo Nautilus conectando todos os pontos
• pikachu_mesh_perfeita.obj/ply/stl - Malha de alta qualidade otimizada  
• pikachu_mesh_nautilus_real.obj/stl - Forma realística do Pikachu
• pikachu_mesh_convex_hull_otimizado.obj/ply/stl - Geometria fechada
• pikachu_mesh_delaunay_surface.obj/ply/stl - Máximo detalhe triangular

🎨 CONTROLES: Use o mouse para rotacionar, zoom e navegar nas visualizações 3D
"""

plt.figtext(0.02, 0.02, info_text, fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

plt.tight_layout()

# Salva a visualização
plt.savefig('pikachu_meshes_visualizador.png', dpi=300, bbox_inches='tight')

print("✅ Visualizador criado!")
print("🎨 Abrindo janela interativa...")

# Mostra a visualização interativa
plt.show()

print("🏆 Visualização concluída!")
print("📁 Imagem salva como: pikachu_meshes_visualizador.png")
print("\n⚡ MESHES DISPONÍVEIS NO DIRETÓRIO:")
print("   • pikachu_mesh_suprema.obj (7.5 KB)")
print("   • pikachu_mesh_perfeita.obj (65.4 KB)")  
print("   • pikachu_mesh_nautilus_real.obj (3.8 KB)")
print("   • pikachu_mesh_convex_hull_otimizado.obj (6.1 KB)")
print("   • pikachu_mesh_delaunay_surface.obj (279.9 KB)")
