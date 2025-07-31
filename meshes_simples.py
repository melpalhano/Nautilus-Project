import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import os

print("🔥 INICIANDO GERAÇÃO DE MESHES...")

# Carrega point cloud
print("📂 Carregando point cloud...")
pc = trimesh.load("pikachu_pointcloud_nautilus.ply")
pontos = pc.vertices
print(f"✅ {len(pontos)} pontos carregados")

# Prepara pontos
print("🔧 Preparando pontos...")
pontos = pontos[~np.isnan(pontos).any(axis=1)]
centroid = np.mean(pontos, axis=0)
pontos = pontos - centroid
max_dist = np.max(np.linalg.norm(pontos, axis=1))
pontos = pontos / max_dist
print(f"✅ {len(pontos)} pontos processados")

# Gera meshes
print("🎯 Gerando meshes...")

# Mesh 1: ConvexHull simples
hull1 = ConvexHull(pontos)
mesh1 = trimesh.Trimesh(vertices=pontos, faces=hull1.simplices)
mesh1.remove_duplicate_faces()
mesh1.fix_normals()
print(f"✅ Mesh 1: {len(mesh1.vertices)}v, {len(mesh1.faces)}f")

# Mesh 2: ConvexHull com amostragem
indices = np.random.choice(len(pontos), min(800, len(pontos)), replace=False)
pontos_sample = pontos[indices]
hull2 = ConvexHull(pontos_sample)
mesh2 = trimesh.Trimesh(vertices=pontos_sample, faces=hull2.simplices)
mesh2.remove_duplicate_faces()
mesh2.fix_normals()
print(f"✅ Mesh 2: {len(mesh2.vertices)}v, {len(mesh2.faces)}f")

# Mesh 3: ConvexHull com menos pontos
indices3 = np.random.choice(len(pontos), min(400, len(pontos)), replace=False)
pontos_sample3 = pontos[indices3]
hull3 = ConvexHull(pontos_sample3)
mesh3 = trimesh.Trimesh(vertices=pontos_sample3, faces=hull3.simplices)
mesh3.remove_duplicate_faces()
mesh3.fix_normals()
print(f"✅ Mesh 3: {len(mesh3.vertices)}v, {len(mesh3.faces)}f")

meshes = [
    ("Alta Densidade", mesh1),
    ("Média Densidade", mesh2),
    ("Baixa Densidade", mesh3)
]

# Salva meshes
print("💾 Salvando meshes...")
for nome, mesh in meshes:
    filename = f"pikachu_mesh_{nome.lower().replace(' ', '_')}"
    mesh.export(f"{filename}.obj")
    print(f"   💎 {filename}.obj salvo")

# Plot SOMENTE meshes
print("🎨 Plotando SOMENTE meshes...")
fig = plt.figure(figsize=(18, 6))
fig.suptitle('🎯 ANÁLISE DE QUALIDADE - ESCOLHA O MELHOR MESH! 🎯', fontsize=16, fontweight='bold')

for i, (nome, mesh) in enumerate(meshes):
    ax = plt.subplot(1, 3, i + 1, projection='3d')
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Plota mesh com faces
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   triangles=faces, cmap='plasma', alpha=0.9,
                   edgecolor='darkred', linewidth=0.1)
    
    # Título com qualidade
    watertight = '🌊 PERFEITA' if mesh.is_watertight else '⚠️ COM BURACOS'
    title = f'🏆 {nome}\n{len(vertices):,}v, {len(faces):,}f\n{watertight}\nÁrea: {mesh.area:.4f}'
    if mesh.is_watertight:
        title += f'\nVolume: {mesh.volume:.4f}'
    
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig('pikachu_meshes_comparacao.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Comparação salva: pikachu_meshes_comparacao.png")
print("🎯 COMPARE OS MESHES E ME DIGA QUAL FICOU MELHOR!")
print("🔥 PROCESSO CONCLUÍDO!")
