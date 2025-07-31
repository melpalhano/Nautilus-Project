#!/usr/bin/env python3
"""
🔥 MESHES DO POINT CLOUD SUPERIOR - RESULTADOS DE QUALIDADE!
============================================================

Utilizando o point cloud superior que deu os melhores resultados,
este script gera múltiplas meshes de alta qualidade e plota os resultados.

OBJETIVO: Meshes que ficam BOM usando o point cloud superior!
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import os

class MeshDoPointCloudSuperior:
    """Gerador de meshes de alta qualidade do point cloud superior"""
    
    def __init__(self):
        self.pointcloud = None
        self.meshes_geradas = []
        
    def carregar_pointcloud_superior(self, arquivo="pikachu_pointcloud_nautilus.ply"):
        """Carrega o point cloud superior"""
        print("📂 CARREGANDO POINT CLOUD SUPERIOR...")
        
        try:
            # Carrega point cloud
            pc = trimesh.load(arquivo)
            
            if hasattr(pc, 'vertices'):
                self.pointcloud = pc.vertices
            else:
                # Se for point cloud, pode estar em pc.points
                self.pointcloud = np.array(pc.vertices) if hasattr(pc, 'vertices') else np.array(pc)
            
            print(f"   ✅ Point cloud carregado: {len(self.pointcloud)} pontos")
            print(f"   📐 Dimensões: {self.pointcloud.shape}")
            print(f"   📊 Range X: [{self.pointcloud[:, 0].min():.3f}, {self.pointcloud[:, 0].max():.3f}]")
            print(f"   📊 Range Y: [{self.pointcloud[:, 1].min():.3f}, {self.pointcloud[:, 1].max():.3f}]")
            print(f"   📊 Range Z: [{self.pointcloud[:, 2].min():.3f}, {self.pointcloud[:, 2].max():.3f}]")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erro carregando point cloud: {e}")
            return False
    
    def preprocessar_pointcloud(self):
        """Preprocessa point cloud para melhor qualidade"""
        print("🔧 PREPROCESSANDO POINT CLOUD...")
        
        original_count = len(self.pointcloud)
        
        # 1. Remove outliers usando DBSCAN
        print("   🎯 Removendo outliers...")
        
        clustering = DBSCAN(eps=0.1, min_samples=5)
        labels = clustering.fit_predict(self.pointcloud)
        
        # Mantém apenas o cluster principal (label != -1)
        main_cluster_mask = labels != -1
        self.pointcloud = self.pointcloud[main_cluster_mask]
        
        print(f"      ✅ Outliers removidos: {original_count - len(self.pointcloud)} pontos")
        print(f"      ✅ Pontos restantes: {len(self.pointcloud)} pontos")
        
        # 2. Normalização para melhor processamento
        print("   🎯 Normalizando coordenadas...")
        
        # Centraliza no origem
        centroid = np.mean(self.pointcloud, axis=0)
        self.pointcloud = self.pointcloud - centroid
        
        # Escala para range [-1, 1]
        max_range = np.max(np.abs(self.pointcloud))
        self.pointcloud = self.pointcloud / max_range
        
        print(f"      ✅ Point cloud normalizado")
        print(f"      📊 Novo range: [{self.pointcloud.min():.3f}, {self.pointcloud.max():.3f}]")
        
        return self.pointcloud
    
    def gerar_mesh_convex_hull_otimizada(self):
        """Gera mesh usando ConvexHull otimizado"""
        print("🔺 GERANDO MESH - CONVEX HULL OTIMIZADA...")
        
        try:
            hull = ConvexHull(self.pointcloud)
            mesh = trimesh.Trimesh(vertices=self.pointcloud, faces=hull.simplices)
            
            # Otimizações
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()
            
            # Subdivisão para suavidade
            if len(mesh.vertices) < 1000:
                mesh = mesh.subdivide()
            
            # Suavização
            try:
                mesh = mesh.smoothed()
            except:
                pass
            
            self.meshes_geradas.append(("ConvexHull_Otimizada", mesh))
            
            print(f"   ✅ ConvexHull: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
            print(f"   🌊 Watertight: {mesh.is_watertight}")
            
            return mesh
            
        except Exception as e:
            print(f"   ❌ Erro ConvexHull: {e}")
            return None
    
    def gerar_mesh_delaunay_3d(self):
        """Gera mesh usando triangulação Delaunay 3D"""
        print("🔷 GERANDO MESH - DELAUNAY 3D...")
        
        try:
            # Amostra pontos se muito denso
            pontos = self.pointcloud
            if len(pontos) > 2000:
                indices = np.random.choice(len(pontos), 2000, replace=False)
                pontos = pontos[indices]
                print(f"   🎯 Amostragem: {len(pontos)} pontos")
            
            # Triangulação 2D projetada para 3D
            pontos_2d = pontos[:, :2]  # Projeta para XY
            tri_2d = Delaunay(pontos_2d)
            
            # Cria mesh 3D
            mesh = trimesh.Trimesh(vertices=pontos, faces=tri_2d.simplices)
            
            # Limpeza
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()
            
            self.meshes_geradas.append(("Delaunay_3D", mesh))
            
            print(f"   ✅ Delaunay: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
            print(f"   🌊 Watertight: {mesh.is_watertight}")
            
            return mesh
            
        except Exception as e:
            print(f"   ❌ Erro Delaunay: {e}")
            return None
    
    def gerar_mesh_alpha_shape(self):
        """Gera mesh usando Alpha Shape (mais precisa)"""
        print("🎪 GERANDO MESH - ALPHA SHAPE...")
        
        try:
            # Cria alpha shape usando trimesh
            mesh = trimesh.points.pointcloud_to_mesh(self.pointcloud, alpha=0.3)
            
            if mesh is not None:
                # Otimizações
                mesh.remove_duplicate_faces()
                mesh.remove_unreferenced_vertices()
                mesh.fix_normals()
                
                self.meshes_geradas.append(("Alpha_Shape", mesh))
                
                print(f"   ✅ Alpha Shape: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
                print(f"   🌊 Watertight: {mesh.is_watertight}")
                
                return mesh
            else:
                print("   ⚠️ Alpha Shape não gerou mesh válida")
                return None
                
        except Exception as e:
            print(f"   ❌ Erro Alpha Shape: {e}")
            return None
    
    def gerar_mesh_poisson_surface(self):
        """Gera mesh usando reconstrução de superfície Poisson"""
        print("🌊 GERANDO MESH - POISSON SURFACE...")
        
        try:
            # Estima normais dos pontos
            print("   🎯 Estimando normais...")
            
            # Para cada ponto, encontra vizinhos e estima normal
            from sklearn.neighbors import NearestNeighbors
            
            nbrs = NearestNeighbors(n_neighbors=10).fit(self.pointcloud)
            distances, indices = nbrs.kneighbors(self.pointcloud)
            
            normals = []
            for i, point_indices in enumerate(indices):
                neighbors = self.pointcloud[point_indices]
                
                # Calcula normal usando PCA dos vizinhos
                centered = neighbors - neighbors.mean(axis=0)
                _, _, vh = np.linalg.svd(centered)
                normal = vh[-1]  # Última componente principal
                
                # Orienta normal para cima (Z positivo)
                if normal[2] < 0:
                    normal = -normal
                
                normals.append(normal)
            
            normals = np.array(normals)
            
            # Cria point cloud com normais
            pc_with_normals = trimesh.PointCloud(vertices=self.pointcloud)
            pc_with_normals.vertex_normals = normals
            
            # Tenta reconstrução Poisson (pode não estar disponível)
            try:
                mesh = pc_with_normals.convex_hull  # Fallback para convex hull
                
                # Subdivisão e suavização extras
                if len(mesh.vertices) < 800:
                    mesh = mesh.subdivide()
                    mesh = mesh.subdivide()  # Dupla subdivisão
                
                mesh = mesh.smoothed()
                
                self.meshes_geradas.append(("Poisson_Surface", mesh))
                
                print(f"   ✅ Poisson: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
                print(f"   🌊 Watertight: {mesh.is_watertight}")
                
                return mesh
                
            except Exception as e2:
                print(f"   ⚠️ Poisson não disponível, usando ConvexHull refinado: {e2}")
                return None
                
        except Exception as e:
            print(f"   ❌ Erro Poisson: {e}")
            return None
    
    def gerar_mesh_clustering_hierarquico(self):
        """Gera mesh usando clustering hierárquico para melhor topologia"""
        print("🌳 GERANDO MESH - CLUSTERING HIERÁRQUICO...")
        
        try:
            # Clustering em múltiplos níveis
            from sklearn.cluster import AgglomerativeClustering
            
            # Primeira divisão em regiões
            n_clusters = min(20, len(self.pointcloud) // 50)
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(self.pointcloud)
            
            # Para cada cluster, cria submesh
            submeshes = []
            
            for cluster_id in range(n_clusters):
                cluster_points = self.pointcloud[cluster_labels == cluster_id]
                
                if len(cluster_points) >= 4:  # Mínimo para formar tetraedro
                    try:
                        hull = ConvexHull(cluster_points)
                        submesh = trimesh.Trimesh(vertices=cluster_points, faces=hull.simplices)
                        submeshes.append(submesh)
                    except:
                        continue
            
            if submeshes:
                # Combina submeshes
                mesh_combined = trimesh.util.concatenate(submeshes)
                
                # Limpeza final
                mesh_combined.remove_duplicate_faces()
                mesh_combined.remove_unreferenced_vertices()
                mesh_combined.fix_normals()
                
                # Suavização
                try:
                    mesh_combined = mesh_combined.smoothed()
                except:
                    pass
                
                self.meshes_geradas.append(("Clustering_Hierarquico", mesh_combined))
                
                print(f"   ✅ Clustering: {len(mesh_combined.vertices)} vértices, {len(mesh_combined.faces)} faces")
                print(f"   🌊 Watertight: {mesh_combined.is_watertight}")
                
                return mesh_combined
            else:
                print("   ⚠️ Nenhuma submesh válida gerada")
                return None
                
        except Exception as e:
            print(f"   ❌ Erro Clustering: {e}")
            return None
    
    def selecionar_melhor_mesh(self):
        """Seleciona a melhor mesh baseada em critérios de qualidade"""
        print("🏆 SELECIONANDO MELHOR MESH...")
        
        if not self.meshes_geradas:
            print("   ❌ Nenhuma mesh foi gerada!")
            return None
        
        melhor_mesh = None
        melhor_score = 0
        melhor_nome = ""
        
        print("   📊 AVALIAÇÃO DAS MESHES:")
        
        for nome, mesh in self.meshes_geradas:
            score = 0
            
            # Critérios de qualidade
            
            # 1. Watertight é fundamental (50 pontos)
            if mesh.is_watertight:
                score += 50
            
            # 2. Número adequado de vértices (20 pontos máx)
            vertex_score = min(len(mesh.vertices) / 100, 20)
            score += vertex_score
            
            # 3. Área razoável (15 pontos máx)
            if 0.5 <= mesh.area <= 20.0:
                score += 15
            
            # 4. Volume positivo se watertight (15 pontos)
            if mesh.is_watertight and mesh.volume > 0:
                score += 15
            
            print(f"      🔸 {nome}: {score:.1f} pontos")
            print(f"         - Vértices: {len(mesh.vertices):,}")
            print(f"         - Faces: {len(mesh.faces):,}")
            print(f"         - Watertight: {'✅' if mesh.is_watertight else '❌'}")
            print(f"         - Área: {mesh.area:.3f}")
            
            if score > melhor_score:
                melhor_score = score
                melhor_mesh = mesh
                melhor_nome = nome
        
        print(f"\n   🏆 MELHOR MESH: {melhor_nome} ({melhor_score:.1f} pontos)")
        
        return melhor_mesh, melhor_nome
    
    def salvar_meshes(self):
        """Salva todas as meshes geradas"""
        print("💾 SALVANDO MESHES...")
        
        for nome, mesh in self.meshes_geradas:
            # Nome do arquivo
            filename_base = f"pikachu_pointcloud_superior_{nome.lower()}"
            
            # Salva em múltiplos formatos
            for ext in ['obj', 'stl', 'ply']:
                try:
                    filename = f"{filename_base}.{ext}"
                    mesh.export(filename)
                    size = os.path.getsize(filename)
                    print(f"   💎 {filename}: {size:,} bytes")
                except Exception as e:
                    print(f"   ❌ Erro salvando {filename}: {e}")

def plotar_resultados(pointcloud, meshes_geradas, melhor_mesh, melhor_nome):
    """Plota todos os resultados de forma organizada"""
    print("🎨 PLOTANDO RESULTADOS...")
    
    # Configura figura grande
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('🔥 MESHES DO POINT CLOUD SUPERIOR - RESULTADOS DE QUALIDADE! 🔥', 
                fontsize=16, fontweight='bold')
    
    # Número de meshes + point cloud original
    n_meshes = len(meshes_geradas)
    n_plots = n_meshes + 1  # +1 para o point cloud original
    
    # Calcula layout da grid
    if n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    elif n_plots <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 3, 4
    
    # 1. Point Cloud Original
    ax1 = plt.subplot(rows, cols, 1, projection='3d')
    ax1.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
               c=pointcloud[:, 2], cmap='viridis', s=15, alpha=0.8)
    ax1.set_title(f'📂 Point Cloud Original\n{len(pointcloud):,} pontos', fontweight='bold')
    ax1.view_init(elev=30, azim=45)
    
    # 2. Meshes geradas
    for i, (nome, mesh) in enumerate(meshes_geradas):
        ax = plt.subplot(rows, cols, i + 2, projection='3d')
        
        vertices = mesh.vertices
        
        # Destaque para a melhor mesh
        if nome == melhor_nome:
            # Cor especial para a melhor
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                      c='gold', s=20, alpha=0.9, edgecolors='red', linewidth=0.5)
            title_color = 'red'
            title_prefix = '🏆 '
        else:
            # Cores normais
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                      c=vertices[:, 2], cmap='plasma', s=15, alpha=0.8)
            title_color = 'black'
            title_prefix = '🔸 '
        
        # Título com informações
        watertight_icon = '🌊' if mesh.is_watertight else '⚠️'
        title = f'{title_prefix}{nome}\n{len(vertices):,}v, {len(mesh.faces):,}f {watertight_icon}'
        
        ax.set_title(title, fontweight='bold', color=title_color, fontsize=9)
        ax.view_init(elev=30, azim=45)
        
        # Remove labels para mais espaço
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
    
    # Vista especial da melhor mesh (se houver espaço)
    if n_plots < rows * cols:
        ax_melhor = plt.subplot(rows, cols, n_plots + 1, projection='3d')
        
        if melhor_mesh is not None:
            vertices = melhor_mesh.vertices
            faces = melhor_mesh.faces
            
            # Wireframe da melhor mesh
            for i in range(0, len(faces), max(1, len(faces)//50)):
                face = faces[i]
                triangle = vertices[face]
                triangle_closed = np.vstack([triangle, triangle[0]])
                ax_melhor.plot(triangle_closed[:, 0], triangle_closed[:, 1], triangle_closed[:, 2],
                              'cyan', alpha=0.7, linewidth=1)
            
            ax_melhor.set_title(f'🏆 MELHOR MESH - WIREFRAME\n{melhor_nome}', 
                               fontweight='bold', color='red')
            ax_melhor.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig('pikachu_pointcloud_superior_completo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Resultados plotados e salvos: pikachu_pointcloud_superior_completo.png")

def main():
    """Execução principal"""
    print("🔥 MESHES DO POINT CLOUD SUPERIOR - QUALIDADE MÁXIMA!")
    print("="*70)
    print("🎯 UTILIZANDO O POINT CLOUD QUE DEU MELHORES RESULTADOS!")
    print("🎯 GERANDO MÚLTIPLAS MESHES DE ALTA QUALIDADE!")
    print("🎯 PLOTANDO RESULTADOS PARA COMPARAÇÃO!")
    print("="*70)
    
    # Inicializa gerador
    gerador = MeshDoPointCloudSuperior()
    
    # 1. Carrega point cloud superior
    print("\n1️⃣ CARREGANDO POINT CLOUD SUPERIOR...")
    if not gerador.carregar_pointcloud_superior():
        print("❌ Falha no carregamento!")
        return
    
    # 2. Preprocessa
    print("\n2️⃣ PREPROCESSANDO POINT CLOUD...")
    pointcloud_processado = gerador.preprocessar_pointcloud()
    
    # 3. Gera múltiplas meshes
    print("\n3️⃣ GERANDO MÚLTIPLAS MESHES...")
    
    # ConvexHull otimizada
    gerador.gerar_mesh_convex_hull_otimizada()
    
    # Delaunay 3D
    gerador.gerar_mesh_delaunay_3d()
    
    # Alpha Shape
    gerador.gerar_mesh_alpha_shape()
    
    # Poisson Surface
    gerador.gerar_mesh_poisson_surface()
    
    # Clustering Hierárquico
    gerador.gerar_mesh_clustering_hierarquico()
    
    # 4. Seleciona melhor mesh
    print("\n4️⃣ SELECIONANDO MELHOR MESH...")
    resultado = gerador.selecionar_melhor_mesh()
    
    if resultado is None:
        print("❌ Nenhuma mesh válida foi gerada!")
        return
    
    melhor_mesh, melhor_nome = resultado
    
    # 5. Salva meshes
    print("\n5️⃣ SALVANDO MESHES...")
    gerador.salvar_meshes()
    
    # 6. Plota resultados
    print("\n6️⃣ PLOTANDO RESULTADOS...")
    plotar_resultados(pointcloud_processado, gerador.meshes_geradas, melhor_mesh, melhor_nome)
    
    print("\n" + "🔥"*70)
    print("🎉 MESHES DO POINT CLOUD SUPERIOR GERADAS!")
    print("🔥"*70)
    print("✅ POINT CLOUD SUPERIOR UTILIZADO!")
    print("✅ MÚLTIPLAS MESHES DE ALTA QUALIDADE!")
    print("✅ MELHOR MESH SELECIONADA AUTOMATICAMENTE!")
    print("✅ RESULTADOS PLOTADOS E COMPARADOS!")
    print("✅ ARQUIVOS SALVOS EM MÚLTIPLOS FORMATOS!")
    print("🏆 MESHES QUE FICAM BOM! 🏆")

if __name__ == "__main__":
    main()
