#!/usr/bin/env python3
"""
🔥 MESHES DO POINT CLOUD SUPERIOR - VERSÃO SIMPLES E ROBUSTA
============================================================

Pega o point cloud superior e gera meshes que ficam BOM!
Versão simplificada e testada.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay
import os

def carregar_pointcloud():
    """Carrega o point cloud superior"""
    print("📂 CARREGANDO POINT CLOUD SUPERIOR...")
    
    try:
        # Tenta carregar o point cloud
        pc = trimesh.load("pikachu_pointcloud_nautilus.ply")
        
        if hasattr(pc, 'vertices'):
            pontos = pc.vertices
        else:
            pontos = np.array(pc)
        
        print(f"   ✅ Point cloud carregado: {len(pontos)} pontos")
        print(f"   📊 Shape: {pontos.shape}")
        
        return pontos
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def preprocessar_pontos(pontos):
    """Limpa e prepara os pontos"""
    print("🔧 PREPARANDO PONTOS...")
    
    # Remove NaN e infinitos
    pontos = pontos[~np.isnan(pontos).any(axis=1)]
    pontos = pontos[~np.isinf(pontos).any(axis=1)]
    
    # Centraliza
    centroid = np.mean(pontos, axis=0)
    pontos = pontos - centroid
    
    # Normaliza
    max_dist = np.max(np.linalg.norm(pontos, axis=1))
    pontos = pontos / max_dist
    
    print(f"   ✅ Pontos processados: {len(pontos)}")
    print(f"   📊 Range: [{pontos.min():.3f}, {pontos.max():.3f}]")
    
    return pontos

def gerar_mesh_convexhull(pontos):
    """Gera mesh usando ConvexHull"""
    print("🔺 MESH 1: ConvexHull...")
    
    try:
        hull = ConvexHull(pontos)
        mesh = trimesh.Trimesh(vertices=pontos, faces=hull.simplices)
        
        # Limpeza básica
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        
        print(f"   ✅ ConvexHull: {len(mesh.vertices)}v, {len(mesh.faces)}f")
        
        return mesh
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def gerar_mesh_pikachu_shape(pontos):
    """Gera mesh que preserva a forma do Pikachu com buracos estratégicos"""
    print("⚡ MESH 2: Pikachu Shape...")
    
    try:
        from sklearn.cluster import DBSCAN
        from scipy.spatial.distance import pdist, squareform
        
        # 1. Clustering para identificar partes do corpo
        clustering = DBSCAN(eps=0.1, min_samples=10).fit(pontos)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        print(f"   🔍 Encontradas {n_clusters} partes do corpo")
        
        # 2. Processa cada cluster separadamente
        meshes_partes = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # Ignora ruído
                continue
                
            pontos_cluster = pontos[labels == cluster_id]
            
            if len(pontos_cluster) > 10:  # Só processa clusters significativos
                try:
                    # Delaunay 3D para preservar formas complexas
                    from scipy.spatial import Delaunay as Delaunay3D
                    tri = Delaunay3D(pontos_cluster)
                    mesh_parte = trimesh.Trimesh(vertices=pontos_cluster, faces=tri.simplices)
                    meshes_partes.append(mesh_parte)
                except:
                    # Fallback para ConvexHull se Delaunay falhar
                    if len(pontos_cluster) > 4:
                        hull = ConvexHull(pontos_cluster)
                        mesh_parte = trimesh.Trimesh(vertices=pontos_cluster, faces=hull.simplices)
                        meshes_partes.append(mesh_parte)
        
        # 3. Combina todas as partes
        if meshes_partes:
            mesh_final = meshes_partes[0]
            for mesh_parte in meshes_partes[1:]:
                try:
                    mesh_final = mesh_final.union(mesh_parte)
                except:
                    # Se união falhar, concatena
                    vertices = np.vstack([mesh_final.vertices, mesh_parte.vertices])
                    faces_offset = len(mesh_final.vertices)
                    faces = np.vstack([mesh_final.faces, mesh_parte.faces + faces_offset])
                    mesh_final = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            # Fallback: Delaunay simples
            tri = Delaunay(pontos[:, :2])
            mesh_final = trimesh.Trimesh(vertices=pontos, faces=tri.simplices)
        
        # 4. Limpeza preservando a forma
        mesh_final.remove_duplicate_faces()
        mesh_final.remove_unreferenced_vertices()
        # NÃO fazer fix_normals para preservar buracos intencionais
        
        print(f"   ✅ Pikachu Shape: {len(mesh_final.vertices)}v, {len(mesh_final.faces)}f")
        
        return mesh_final
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        # Fallback seguro
        try:
            tri = Delaunay(pontos[:, :2])
            mesh = trimesh.Trimesh(vertices=pontos, faces=tri.simplices)
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            return mesh
        except:
            return None

def gerar_mesh_orelhas_separadas(pontos):
    """Gera mesh preservando orelhas e características do Pikachu"""
    print("👂 MESH 3: Orelhas Separadas...")
    
    try:
        # 1. Identifica regiões altas (orelhas) e baixas (corpo)
        z_coords = pontos[:, 2]
        z_threshold_high = np.percentile(z_coords, 85)  # Top 15% para orelhas
        z_threshold_low = np.percentile(z_coords, 15)   # Bottom 15% para base
        
        # 2. Separa pontos por regiões
        pontos_orelhas = pontos[z_coords > z_threshold_high]
        pontos_corpo = pontos[(z_coords >= z_threshold_low) & (z_coords <= z_threshold_high)]
        pontos_base = pontos[z_coords < z_threshold_low]
        
        print(f"   👂 Orelhas: {len(pontos_orelhas)} pontos")
        print(f"   🎯 Corpo: {len(pontos_corpo)} pontos") 
        print(f"   🔽 Base: {len(pontos_base)} pontos")
        
        meshes_regioes = []
        
        # 3. Mesh das orelhas (preserva picos)
        if len(pontos_orelhas) > 4:
            try:
                hull_orelhas = ConvexHull(pontos_orelhas)
                mesh_orelhas = trimesh.Trimesh(vertices=pontos_orelhas, faces=hull_orelhas.simplices)
                meshes_regioes.append(mesh_orelhas)
            except:
                pass
        
        # 4. Mesh do corpo (com mais detalhes)
        if len(pontos_corpo) > 10:
            try:
                # Amostra pontos do corpo para Delaunay
                if len(pontos_corpo) > 800:
                    indices = np.random.choice(len(pontos_corpo), 800, replace=False)
                    pontos_corpo_sample = pontos_corpo[indices]
                else:
                    pontos_corpo_sample = pontos_corpo
                
                tri_corpo = Delaunay(pontos_corpo_sample[:, :2])
                mesh_corpo = trimesh.Trimesh(vertices=pontos_corpo_sample, faces=tri_corpo.simplices)
                meshes_regioes.append(mesh_corpo)
            except:
                # Fallback para ConvexHull
                if len(pontos_corpo) > 4:
                    hull_corpo = ConvexHull(pontos_corpo)
                    mesh_corpo = trimesh.Trimesh(vertices=pontos_corpo, faces=hull_corpo.simplices)
                    meshes_regioes.append(mesh_corpo)
        
        # 5. Mesh da base
        if len(pontos_base) > 4:
            try:
                hull_base = ConvexHull(pontos_base)
                mesh_base = trimesh.Trimesh(vertices=pontos_base, faces=hull_base.simplices)
                meshes_regioes.append(mesh_base)
            except:
                pass
        
        # 6. Combina todas as regiões
        if meshes_regioes:
            mesh_final = meshes_regioes[0]
            for mesh_regiao in meshes_regioes[1:]:
                try:
                    # Concatena vertices e faces
                    vertices_combined = np.vstack([mesh_final.vertices, mesh_regiao.vertices])
                    faces_offset = len(mesh_final.vertices)
                    faces_combined = np.vstack([mesh_final.faces, mesh_regiao.faces + faces_offset])
                    mesh_final = trimesh.Trimesh(vertices=vertices_combined, faces=faces_combined)
                except:
                    pass
        else:
            # Fallback para todos os pontos
            hull = ConvexHull(pontos)
            mesh_final = trimesh.Trimesh(vertices=pontos, faces=hull.simplices)
        
        # 7. Limpeza mínima (preserva características)
        mesh_final.remove_duplicate_faces()
        mesh_final.remove_unreferenced_vertices()
        
        print(f"   ✅ Orelhas Separadas: {len(mesh_final.vertices)}v, {len(mesh_final.faces)}f")
        
        return mesh_final
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def gerar_mesh_baixa_resolucao(pontos):
    """Gera mesh de baixa resolução para performance"""
    print("� MESH 5: Baixa Resolução...")
    
    try:
        # Amostra poucos pontos
        if len(pontos) > 500:
            indices = np.random.choice(len(pontos), 500, replace=False)
            pontos_sample = pontos[indices]
        else:
            pontos_sample = pontos
        
        hull = ConvexHull(pontos_sample)
        mesh = trimesh.Trimesh(vertices=pontos_sample, faces=hull.simplices)
        
        # Limpeza
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        
        print(f"   ✅ Baixa Res: {len(mesh.vertices)}v, {len(mesh.faces)}f")
        
        return mesh
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def gerar_mesh_alta_resolucao(pontos):
    """Gera mesh de alta resolução para detalhes"""
    print("🔺 MESH 5: Alta Resolução...")
    
    try:
        # Usa todos os pontos disponíveis
        hull = ConvexHull(pontos)
        mesh = trimesh.Trimesh(vertices=pontos, faces=hull.simplices)
        
        # Subdivisão para mais detalhes
        try:
            mesh = mesh.subdivide()
        except:
            pass
        
        # Limpeza
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        
        print(f"   ✅ Alta Res: {len(mesh.vertices)}v, {len(mesh.faces)}f")
        
        return mesh
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def gerar_mesh_anatomia_pikachu(pontos):
    """Gera mesh respeitando a anatomia do Pikachu com buracos estratégicos"""
    print("⚡ MESH 6: Anatomia Pikachu...")
    
    try:
        from sklearn.cluster import DBSCAN
        
        # 1. Clustering para identificar partes anatômicas
        clustering = DBSCAN(eps=0.08, min_samples=8).fit(pontos)
        labels = clustering.labels_
        
        # 2. Processa clusters significativos
        clusters_validos = []
        for cluster_id in set(labels):
            if cluster_id != -1:  # Ignora ruído
                pontos_cluster = pontos[labels == cluster_id]
                if len(pontos_cluster) > 15:  # Só clusters significativos
                    clusters_validos.append(pontos_cluster)
        
        print(f"   🧩 {len(clusters_validos)} partes anatômicas identificadas")
        
        # 3. Gera mesh para cada parte usando Delaunay (preserva buracos)
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for i, cluster_pontos in enumerate(clusters_validos):
            try:
                # Delaunay 2D para preservar forma natural
                tri = Delaunay(cluster_pontos[:, :2])
                
                # Adiciona vértices
                all_vertices.extend(cluster_pontos)
                
                # Adiciona faces com offset
                faces_cluster = tri.simplices + vertex_offset
                all_faces.extend(faces_cluster)
                vertex_offset += len(cluster_pontos)
                
            except Exception as e:
                print(f"   ⚠️  Cluster {i} falhou: {e}")
                continue
        
        # 4. Cria mesh final
        if all_vertices and all_faces:
            mesh_final = trimesh.Trimesh(vertices=np.array(all_vertices), faces=np.array(all_faces))
            
            # Limpeza mínima (preserva buracos característicos)
            mesh_final.remove_duplicate_faces()
            mesh_final.remove_unreferenced_vertices()
            # NÃO usar fix_normals para manter buracos
            
            print(f"   ✅ Anatomia Pikachu: {len(mesh_final.vertices)}v, {len(mesh_final.faces)}f")
            return mesh_final
        else:
            raise Exception("Nenhuma parte anatômica válida")
        
    except Exception as e:
        print(f"   ❌ Erro na anatomia: {e}")
        # Fallback: Delaunay simples com buracos
        try:
            if len(pontos) > 1200:
                indices = np.random.choice(len(pontos), 1200, replace=False)
                pontos_sample = pontos[indices]
            else:
                pontos_sample = pontos
                
            tri = Delaunay(pontos_sample[:, :2])
            mesh = trimesh.Trimesh(vertices=pontos_sample, faces=tri.simplices)
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            # Sem fix_normals para preservar características
            return mesh
        except:
            return None

def salvar_meshes(meshes):
    """Salva todas as meshes"""
    print("💾 SALVANDO MESHES...")
    
    for i, (nome, mesh) in enumerate(meshes):
        if mesh is not None:
            filename = f"pikachu_pointcloud_superior_{nome.lower()}"
            
            # Salva em OBJ
            try:
                obj_file = f"{filename}.obj"
                mesh.export(obj_file)
                size = os.path.getsize(obj_file)
                print(f"   💎 {obj_file}: {size:,} bytes")
            except Exception as e:
                print(f"   ❌ Erro salvando {filename}: {e}")

def plotar_meshes(pontos_original, meshes):
    """Plota SOMENTE os meshes (sem point cloud) para análise de qualidade"""
    print("🎨 PLOTANDO SOMENTE MESHES PARA ANÁLISE...")
    
    # Filtra meshes válidas
    meshes_validas = [(nome, mesh) for nome, mesh in meshes if mesh is not None]
    
    if not meshes_validas:
        print("❌ Nenhuma mesh válida para plotar!")
        return
    
    n_plots = len(meshes_validas)
    
    # Layout para múltiplas meshes
    if n_plots <= 2:
        rows, cols = 1, 2
        figsize = (16, 8)
    elif n_plots <= 4:
        rows, cols = 2, 2
        figsize = (16, 12)
    else:
        rows, cols = 2, 3
        figsize = (20, 12)
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle('🎯 ANÁLISE DE QUALIDADE DOS MESHES - ESCOLHA O MELHOR! 🎯', 
                fontsize=16, fontweight='bold')
    
    # Plota cada mesh em subplot separado
    for i, (nome, mesh) in enumerate(meshes_validas):
        ax = plt.subplot(rows, cols, i + 1, projection='3d')
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Cor baseada na qualidade e altura
        if mesh.is_watertight:
            # Mesh perfeita - dourado com gradiente
            color = vertices[:, 2]  # baseado na altura Z
            cmap = 'plasma'
            edge_color = 'darkred'
            title_prefix = '🏆 PERFEITA '
            alpha = 1.0
        else:
            # Mesh com problemas - cores mais frias
            color = vertices[:, 2]
            cmap = 'viridis'
            edge_color = 'navy'
            title_prefix = '🔸 '
            alpha = 0.8
        
        # Plot com faces visíveis
        try:
            # Tenta plotar as faces da mesh
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           triangles=faces, cmap=cmap, alpha=alpha,
                           edgecolor=edge_color, linewidth=0.1)
        except:
            # Se falhar, plota como pontos
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                      c=color, cmap=cmap, s=20, alpha=alpha,
                      edgecolors=edge_color, linewidth=0.5)
        
        # Título detalhado com qualidade
        watertight = '🌊 WATERTIGHT' if mesh.is_watertight else '⚠️  COM BURACOS'
        area_info = f"Área: {mesh.area:.4f}"
        volume_info = f"Vol: {mesh.volume:.4f}" if mesh.is_watertight else "Vol: N/A"
        
        title = f'{title_prefix}{nome}\n{len(vertices):,}v, {len(faces):,}f\n{watertight}\n{area_info} | {volume_info}'
        
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.view_init(elev=25, azim=45)
        
        # Remove eixos para foco na mesh
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('pikachu_meshes_qualidade_analise.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Análise salva: pikachu_meshes_qualidade_analise.png")
    print("   🎯 COMPARE AS MESHES E ME DIGA QUAL FICOU MELHOR!")

def main():
    """Execução principal"""
    print("🔥 MESHES DO POINT CLOUD SUPERIOR")
    print("="*50)
    print("🎯 USANDO O POINT CLOUD QUE DEU MELHOR RESULTADO!")
    print("🎯 GERANDO MESHES QUE FICAM BOM!")
    print("="*50)
    
    # 1. Carrega point cloud
    print("\n1️⃣ CARREGANDO...")
    pontos = carregar_pointcloud()
    
    if pontos is None:
        print("❌ Falha no carregamento!")
        return
    
    # 2. Prepara pontos
    print("\n2️⃣ PREPARANDO...")
    pontos_limpos = preprocessar_pontos(pontos)
    
    # 3. Gera meshes
    print("\n3️⃣ GERANDO MÚLTIPLAS MESHES COM FOCO NA FORMA DO PIKACHU...")
    
    mesh1 = gerar_mesh_convexhull(pontos_limpos)
    mesh2 = gerar_mesh_pikachu_shape(pontos_limpos)
    mesh3 = gerar_mesh_orelhas_separadas(pontos_limpos)
    mesh4 = gerar_mesh_baixa_resolucao(pontos_limpos)
    mesh5 = gerar_mesh_alta_resolucao(pontos_limpos)
    mesh6 = gerar_mesh_anatomia_pikachu(pontos_limpos)
    
    meshes = [
        ("ConvexHull Padrão", mesh1),
        ("Pikachu Shape", mesh2),
        ("Orelhas Separadas", mesh3),
        ("Baixa Resolução", mesh4),
        ("Alta Resolução", mesh5),
        ("Anatomia Pikachu", mesh6)
    ]
    
    # 4. Salva
    print("\n4️⃣ SALVANDO...")
    salvar_meshes(meshes)
    
    # 5. Plota
    print("\n5️⃣ PLOTANDO...")
    plotar_meshes(pontos_limpos, meshes)
    
    # 6. Estatísticas finais
    print("\n📊 ESTATÍSTICAS FINAIS:")
    for nome, mesh in meshes:
        if mesh is not None:
            print(f"   🔸 {nome}:")
            print(f"      - Vértices: {len(mesh.vertices):,}")
            print(f"      - Faces: {len(mesh.faces):,}")
            print(f"      - Área: {mesh.area:.6f}")
            print(f"      - Watertight: {'✅' if mesh.is_watertight else '❌'}")
            if mesh.is_watertight:
                print(f"      - Volume: {mesh.volume:.6f}")
    
    print("\n" + "🔥"*50)
    print("🎉 MESHES DO POINT CLOUD SUPERIOR GERADAS!")
    print("🔥"*50)
    print("✅ POINT CLOUD SUPERIOR UTILIZADO!")
    print("✅ MÚLTIPLAS MESHES CRIADAS!")
    print("✅ RESULTADOS PLOTADOS!")
    print("🏆 MESHES QUE FICAM BOM! 🏆")

if __name__ == "__main__":
    main()
