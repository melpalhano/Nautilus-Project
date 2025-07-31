#!/usr/bin/env python3
"""
🔥 NAUTILUS MESH GENERATOR - Pikachu 3D Meshes
==============================================

Gera malhas 3D escaláveis a partir dos point clouds usando:
1. Point Cloud do Nautilus
2. Algoritmos de triangulação avançados
3. Mesh optimization
4. Exportação em múltiplos formatos
5. Visualização interativa das meshes
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from scipy.spatial import ConvexHull, Delaunay
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from PIL import Image
import os

def carregar_pointcloud():
    """Carrega o point cloud do Nautilus"""
    print("☁️ CARREGANDO POINT CLOUD NAUTILUS PARA MESH...")
    
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
        print(f"   ✅ {len(points):,} pontos carregados para geração de mesh")
        return points
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def gerar_mesh_convex_hull(points):
    """Gera mesh usando Convex Hull (método básico)"""
    print("🔺 GERANDO MESH - CONVEX HULL...")
    
    try:
        # Convex Hull para triangulação
        hull = ConvexHull(points)
        
        # Cria mesh
        mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
        
        # Otimizações
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        
        print(f"   ✅ Mesh Convex Hull: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
        return mesh, "convex_hull"
        
    except Exception as e:
        print(f"   ❌ Erro Convex Hull: {e}")
        return None, None

def gerar_mesh_delaunay_3d(points):
    """Gera mesh usando triangulação Delaunay 3D"""
    print("🔷 GERANDO MESH - DELAUNAY 3D...")
    
    try:
        # Triangulação Delaunay 3D
        tri = Delaunay(points)
        
        # Extrai faces da superfície (simplificado)
        faces = []
        for simplex in tri.simplices:
            # Adiciona tetraedros como faces triangulares
            faces.extend([
                [simplex[0], simplex[1], simplex[2]],
                [simplex[0], simplex[1], simplex[3]], 
                [simplex[0], simplex[2], simplex[3]],
                [simplex[1], simplex[2], simplex[3]]
            ])
        
        faces = np.array(faces)
        
        # Cria mesh
        mesh = trimesh.Trimesh(vertices=points, faces=faces)
        
        # Otimizações agressivas
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_faces()
        mesh.fix_normals()
        
        print(f"   ✅ Mesh Delaunay: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
        return mesh, "delaunay_3d"
        
    except Exception as e:
        print(f"   ❌ Erro Delaunay: {e}")
        return None, None

def gerar_mesh_alpha_shape(points):
    """Gera mesh usando Alpha Shape (mais preciso)"""
    print("🔶 GERANDO MESH - ALPHA SHAPE...")
    
    try:
        # Converte para Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estima normais
        pcd.estimate_normals()
        
        # Alpha shape mesh
        alpha = 0.1  # Parâmetro de suavidade
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        
        # Remove outliers
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles() 
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_non_manifold_edges()
        
        # Converte para trimesh
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        
        print(f"   ✅ Mesh Alpha Shape: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
        return mesh, "alpha_shape"
        
    except Exception as e:
        print(f"   ❌ Erro Alpha Shape: {e}")
        return None, None

def gerar_mesh_poisson(points):
    """Gera mesh usando Poisson Surface Reconstruction"""
    print("🌊 GERANDO MESH - POISSON RECONSTRUCTION...")
    
    try:
        # Converte para Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estima normais com orientação consistente
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(30)
        
        # Poisson reconstruction
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove vértices de baixa densidade
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
        
        # Converte para trimesh
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        
        print(f"   ✅ Mesh Poisson: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
        return mesh, "poisson"
        
    except Exception as e:
        print(f"   ❌ Erro Poisson: {e}")
        return None, None

def salvar_mesh(mesh, nome_algoritmo, formato='obj'):
    """Salva mesh em diferentes formatos"""
    if mesh is None:
        return None
        
    nome_arquivo = f'pikachu_mesh_{nome_algoritmo}.{formato}'
    
    try:
        mesh.export(nome_arquivo)
        
        # Informações da mesh
        info = {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'watertight': mesh.is_watertight,
            'volume': mesh.volume if mesh.is_watertight else 'N/A',
            'area': mesh.area,
            'arquivo': nome_arquivo
        }
        
        print(f"   💾 Salvo: {nome_arquivo}")
        print(f"      📊 {info['vertices']} vértices, {info['faces']} faces")
        print(f"      💧 Watertight: {info['watertight']}")
        print(f"      📐 Área: {info['area']:.4f}")
        
        return info
        
    except Exception as e:
        print(f"   ❌ Erro ao salvar: {e}")
        return None

def visualizar_mesh_interativa(mesh, nome_algoritmo):
    """Visualiza mesh de forma interativa"""
    print(f"🎮 VISUALIZANDO MESH {nome_algoritmo.upper()}...")
    
    try:
        # Figura interativa
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot da mesh
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Desenha as faces da mesh
        for face in faces[::10]:  # Mostra apenas algumas faces para performance
            triangle = vertices[face]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                           alpha=0.7, color='red', edgecolor='black', linewidth=0.1)
        
        # Configurações
        ax.set_title(f'🔥 PIKACHU MESH - {nome_algoritmo.upper()} 🔥', 
                    fontsize=14, fontweight='bold')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Vista inicial
        ax.view_init(elev=30, azim=45)
        
        plt.show()
        
    except Exception as e:
        print(f"   ❌ Erro na visualização: {e}")

def gerar_comparacao_meshes(meshes_info):
    """Gera imagem comparativa das meshes"""
    print("📊 GERANDO COMPARAÇÃO DAS MESHES...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': '3d'})
    fig.suptitle('🔥 PIKACHU NAUTILUS - COMPARAÇÃO DE MESHES 🔥', 
                fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (algoritmo, info) in enumerate(meshes_info.items()):
        if i >= 4 or info is None:
            continue
            
        ax = axes[i]
        
        try:
            # Carrega mesh
            mesh = trimesh.load(info['arquivo'])
            vertices = mesh.vertices
            
            # Plot simplificado
            ax.scatter(vertices[::10, 0], vertices[::10, 1], vertices[::10, 2], 
                      s=1, alpha=0.6, c='red')
            
            ax.set_title(f'{algoritmo.replace("_", " ").title()}\n'
                        f'{info["vertices"]} vértices, {info["faces"]} faces', 
                        fontsize=10, fontweight='bold')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            
        except Exception as e:
            ax.text(0.5, 0.5, 0.5, f'Erro: {algoritmo}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('pikachu_meshes_comparacao.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Comparação salva: pikachu_meshes_comparacao.png")

def main():
    """Função principal"""
    print("🔥 NAUTILUS MESH GENERATOR - PIKACHU 3D")
    print("="*60)
    print("🎯 Gerando malhas 3D escaláveis a partir do point cloud")
    print("="*60)
    
    # Carrega point cloud
    points = carregar_pointcloud()
    
    if points is None:
        print("❌ Falha ao carregar point cloud!")
        return
    
    # Dicionário para armazenar informações das meshes
    meshes_info = {}
    
    print("\n🔥 GERANDO MÚLTIPLAS MESHES COM ALGORITMOS DIFERENTES...")
    print("="*60)
    
    # 1. Convex Hull
    mesh1, nome1 = gerar_mesh_convex_hull(points)
    if mesh1:
        info1 = salvar_mesh(mesh1, nome1, 'obj')
        meshes_info[nome1] = info1
        # Salva também em STL
        salvar_mesh(mesh1, nome1, 'stl')
    
    print()
    
    # 2. Delaunay 3D
    mesh2, nome2 = gerar_mesh_delaunay_3d(points)
    if mesh2:
        info2 = salvar_mesh(mesh2, nome2, 'obj')
        meshes_info[nome2] = info2
        salvar_mesh(mesh2, nome2, 'stl')
    
    print()
    
    # 3. Alpha Shape (requer Open3D)
    try:
        mesh3, nome3 = gerar_mesh_alpha_shape(points)
        if mesh3:
            info3 = salvar_mesh(mesh3, nome3, 'obj')
            meshes_info[nome3] = info3
            salvar_mesh(mesh3, nome3, 'stl')
    except ImportError:
        print("⚠️ Open3D não disponível, pulando Alpha Shape...")
    
    print()
    
    # 4. Poisson Reconstruction
    try:
        mesh4, nome4 = gerar_mesh_poisson(points)
        if mesh4:
            info4 = salvar_mesh(mesh4, nome4, 'obj')
            meshes_info[nome4] = info4
            salvar_mesh(mesh4, nome4, 'stl')
    except ImportError:
        print("⚠️ Open3D não disponível, pulando Poisson...")
    
    print("\n" + "="*60)
    print("🎉 GERAÇÃO DE MESHES CONCLUÍDA!")
    print("="*60)
    
    # Resumo
    print("\n📋 RESUMO DAS MESHES GERADAS:")
    for algoritmo, info in meshes_info.items():
        if info:
            print(f"🔸 {algoritmo.replace('_', ' ').title()}:")
            print(f"   📁 Arquivo: {info['arquivo']}")
            print(f"   📊 {info['vertices']} vértices, {info['faces']} faces")
            print(f"   💧 Watertight: {info['watertight']}")
            print()
    
    # Gera comparação visual
    if meshes_info:
        gerar_comparacao_meshes(meshes_info)
    
    print("🔥 NAUTILUS MESHES GERADAS COM SUCESSO! 🔥")
    print("🎮 Use qualquer software 3D para abrir os arquivos .obj ou .stl")
    print("🚀 Meshes escaláveis prontas para impressão 3D ou visualização!")

if __name__ == "__main__":
    main()
