#!/usr/bin/env python3
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay

print("⚡ INICIANDO MESHES PARA FORMA DO PIKACHU...")

# Carrega e prepara pontos
print("📂 Carregando...")
pc = trimesh.load("pikachu_pointcloud_nautilus.ply")
pontos = pc.vertices
pontos = pontos[~np.isnan(pontos).any(axis=1)]
centroid = np.mean(pontos, axis=0)
pontos = pontos - centroid
max_dist = np.max(np.linalg.norm(pontos, axis=1))
pontos = pontos / max_dist
print(f"✅ {len(pontos)} pontos preparados")

# Mesh 1: ConvexHull padrão
print("🔺 Mesh 1: ConvexHull...")
hull1 = ConvexHull(pontos)
mesh1 = trimesh.Trimesh(vertices=pontos, faces=hull1.simplices)
mesh1.remove_duplicate_faces()
mesh1.fix_normals()

# Mesh 2: Delaunay com buracos (SEM fix_normals)
print("⚡ Mesh 2: Delaunay com buracos...")
if len(pontos) > 1000:
    indices = np.random.choice(len(pontos), 1000, replace=False)
    pontos_sample = pontos[indices]
else:
    pontos_sample = pontos

tri = Delaunay(pontos_sample[:, :2])
mesh2 = trimesh.Trimesh(vertices=pontos_sample, faces=tri.simplices)
mesh2.remove_duplicate_faces()
mesh2.remove_unreferenced_vertices()
# SEM mesh2.fix_normals() para preservar buracos!

# Mesh 3: Clustering anatômico
print("👂 Mesh 3: Partes anatômicas...")
try:
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.08, min_samples=8).fit(pontos)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"   {n_clusters} partes encontradas")
    
    # Pega só clusters válidos
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for cluster_id in set(labels):
        if cluster_id != -1:
            cluster_pontos = pontos[labels == cluster_id]
            if len(cluster_pontos) > 15:
                try:
                    tri_cluster = Delaunay(cluster_pontos[:, :2])
                    all_vertices.extend(cluster_pontos)
                    faces_cluster = tri_cluster.simplices + vertex_offset
                    all_faces.extend(faces_cluster)
                    vertex_offset += len(cluster_pontos)
                except:
                    continue
    
    if all_vertices and all_faces:
        mesh3 = trimesh.Trimesh(vertices=np.array(all_vertices), faces=np.array(all_faces))
        mesh3.remove_duplicate_faces()
        mesh3.remove_unreferenced_vertices()
        # SEM fix_normals para preservar anatomia
    else:
        mesh3 = mesh2.copy()
        
except ImportError:
    print("   Sklearn não disponível, usando Delaunay")
    mesh3 = mesh2.copy()

# Mesh 4: Alta resolução
print("🔺 Mesh 4: Alta resolução...")
hull4 = ConvexHull(pontos)
mesh4 = trimesh.Trimesh(vertices=pontos, faces=hull4.simplices)
try:
    mesh4 = mesh4.subdivide()
except:
    pass
mesh4.remove_duplicate_faces()
mesh4.fix_normals()

meshes = [
    ("ConvexHull Padrão", mesh1),
    ("Delaunay COM BURACOS", mesh2),
    ("Anatomia Pikachu", mesh3),
    ("Alta Resolução", mesh4)
]

# Salva meshes
print("💾 Salvando...")
for nome, mesh in meshes:
    filename = f"pikachu_forma_{nome.lower().replace(' ', '_')}"
    mesh.export(f"{filename}.obj")
    print(f"   💎 {filename}.obj")

# Plot focado nos meshes
print("🎨 Plotando meshes com foco na forma do Pikachu...")
fig = plt.figure(figsize=(20, 10))
fig.suptitle('⚡ MESHES COM FOCO NA FORMA DO PIKACHU - ESCOLHA O MELHOR! ⚡', 
            fontsize=16, fontweight='bold')

for i, (nome, mesh) in enumerate(meshes):
    ax = plt.subplot(2, 2, i + 1, projection='3d')
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Cores baseadas na qualidade
    if mesh.is_watertight:
        color_map = 'plasma'
        edge_color = 'darkred'
        title_prefix = '🏆 PERFEITA '
        alpha = 1.0
    else:
        color_map = 'coolwarm'
        edge_color = 'navy'
        title_prefix = '⚡ COM FORMA '
        alpha = 0.9
    
    # Plot da mesh
    try:
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       triangles=faces, cmap=color_map, alpha=alpha,
                       edgecolor=edge_color, linewidth=0.1)
    except:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                  c=vertices[:, 2], cmap=color_map, s=20, alpha=alpha)
    
    # Título detalhado
    watertight = '🌊 FECHADA' if mesh.is_watertight else '⚡ COM BURACOS'
    title = f'{title_prefix}{nome}\n{len(vertices):,}v, {len(faces):,}f\n{watertight}'
    if hasattr(mesh, 'area'):
        title += f'\nÁrea: {mesh.area:.4f}'
    
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig('pikachu_forma_analise.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Análise salva: pikachu_forma_analise.png")
print("⚡ AGORA COMPARE E ME DIGA QUAL TEM MAIS FORMA DE PIKACHU!")
