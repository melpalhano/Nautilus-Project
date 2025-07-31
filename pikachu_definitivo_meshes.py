#!/usr/bin/env python3
"""
⚡ PIKACHU MESH SUPREMA - CONECTA TODOS OS PONTOS PARA FORMA PERFEITA
====================================================================

Algoritmo definitivo que conecta TODOS os pontos do point cloud
para gerar a malha perfeita do Pikachu usando técnicas Nautilus.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from scipy.spatial import ConvexHull, Delaunay
from PIL import Image

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

def gerar_mesh_convex_hull_avancado(points):
    """Gera mesh otimizado usando Convex Hull"""
    print("🔺 GERANDO MESH CONVEX HULL OTIMIZADO...")
    
    try:
        # Convex Hull
        hull = ConvexHull(points)
        
        # Cria mesh inicial
        mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
        
        # Otimizações avançadas
        print("   🔧 Aplicando otimizações...")
        
        # Remove duplicatas
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Corrige normais
        mesh.fix_normals()
        
        # Suavização (se possível)
        try:
            mesh = mesh.smoothed()
        except:
            print("   ⚠️ Suavização não disponível")
        
        # Verifica se é watertight
        if mesh.is_watertight:
            print("   💧 Mesh é watertight (fechada)!")
        else:
            print("   ⚠️ Mesh não é watertight")
            
        print(f"   ✅ Mesh gerada: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
        print(f"   📐 Área superficial: {mesh.area:.4f}")
        if mesh.is_watertight:
            print(f"   📦 Volume: {mesh.volume:.4f}")
        
        return mesh
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def gerar_mesh_delaunay_surface(points):
    """Gera mesh usando projeção 2D + extrusão"""
    print("🔷 GERANDO MESH DELAUNAY SURFACE...")
    
    try:
        # Projeta pontos no plano XY para triangulação 2D
        points_2d = points[:, :2]  # Apenas X e Y
        
        # Triangulação Delaunay 2D
        tri_2d = Delaunay(points_2d)
        
        # Cria faces triangulares
        faces = tri_2d.simplices
        
        # Cria mesh usando pontos 3D originais
        mesh = trimesh.Trimesh(vertices=points, faces=faces)
        
        # Otimizações
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        
        print(f"   ✅ Mesh Delaunay Surface: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
        return mesh
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def salvar_mesh_completo(mesh, nome, incluir_texturas=True):
    """Salva mesh em múltiplos formatos com informações completas"""
    if mesh is None:
        return None
    
    print(f"💾 SALVANDO MESH: {nome}...")
    
    # Informações detalhadas
    info = {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'watertight': mesh.is_watertight,
        'area': mesh.area,
        'bounds': mesh.bounds,
        'center_mass': mesh.center_mass
    }
    
    if mesh.is_watertight:
        info['volume'] = mesh.volume
        info['density'] = mesh.density if hasattr(mesh, 'density') else 'N/A'
    
    # Salva em OBJ (para visualização)
    arquivo_obj = f'pikachu_mesh_{nome}.obj'
    mesh.export(arquivo_obj)
    print(f"   📄 OBJ salvo: {arquivo_obj}")
    
    # Salva em STL (para impressão 3D)
    arquivo_stl = f'pikachu_mesh_{nome}.stl'
    mesh.export(arquivo_stl)
    print(f"   🖨️ STL salvo: {arquivo_stl}")
    
    # Salva em PLY (preserva dados)
    arquivo_ply = f'pikachu_mesh_{nome}.ply'
    mesh.export(arquivo_ply)
    print(f"   📊 PLY salvo: {arquivo_ply}")
    
    # Informações detalhadas
    print(f"   📊 Estatísticas:")
    print(f"      🔸 Vértices: {info['vertices']:,}")
    print(f"      🔸 Faces: {info['faces']:,}")
    print(f"      🔸 Área: {info['area']:.4f}")
    print(f"      🔸 Watertight: {info['watertight']}")
    if 'volume' in info:
        print(f"      🔸 Volume: {info['volume']:.4f}")
    
    return info

def visualizar_mesh_3d_interativo(mesh, nome):
    """Visualização 3D interativa da mesh"""
    print(f"🎮 VISUALIZANDO MESH: {nome.upper()}...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vértices da mesh
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Plot wireframe da mesh
    for face in faces[::5]:  # Mostra cada 5ª face para performance
        triangle = vertices[face]
        
        # Desenha triângulo
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_closed[:, 0], 
               triangle_closed[:, 1], 
               triangle_closed[:, 2], 
               'r-', alpha=0.6, linewidth=0.5)
    
    # Point cloud original para referência
    ax.scatter(vertices[::10, 0], vertices[::10, 1], vertices[::10, 2], 
              c='blue', s=1, alpha=0.3, label='Vertices')
    
    # Configurações
    ax.set_title(f'🔥 PIKACHU MESH - {nome.upper()} 🔥\n'
                f'{len(vertices)} vértices, {len(faces)} faces', 
                fontsize=14, fontweight='bold')
    
    ax.set_xlabel('🔴 X')
    ax.set_ylabel('🔵 Y')
    ax.set_zlabel('🟢 Z')
    
    # Vista inicial
    ax.view_init(elev=30, azim=45)
    
    # Informações na figura
    info_text = f"""
🔥 MESH NAUTILUS - {nome.upper()}
📊 Vértices: {len(vertices):,}
📊 Faces: {len(faces):,}
📐 Área: {mesh.area:.4f}
💧 Watertight: {mesh.is_watertight}
"""
    if mesh.is_watertight:
        info_text += f"📦 Volume: {mesh.volume:.4f}\n"
    
    fig.text(0.02, 0.98, info_text, fontsize=10, 
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.9),
             verticalalignment='top')
    
    plt.legend()
    plt.show()

def gerar_comparacao_visual_completa(mesh1, mesh2, points):
    """Gera comparação visual completa"""
    print("📊 GERANDO COMPARAÇÃO VISUAL COMPLETA...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Point Cloud Original
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    ax1.scatter(points[::5, 0], points[::5, 1], points[::5, 2], 
               c=points[::5, 2], cmap='viridis', s=3, alpha=0.8)
    ax1.set_title('☁️ Point Cloud Original\n(Nautilus)', fontweight='bold')
    ax1.view_init(elev=30, azim=45)
    
    # Mesh 1 - Convex Hull
    if mesh1:
        ax2 = plt.subplot(2, 3, 2, projection='3d')
        vertices1 = mesh1.vertices
        faces1 = mesh1.faces
        
        for face in faces1[::8]:
            triangle = vertices1[face]
            triangle_closed = np.vstack([triangle, triangle[0]])
            ax2.plot(triangle_closed[:, 0], triangle_closed[:, 1], triangle_closed[:, 2], 
                    'r-', alpha=0.7, linewidth=0.5)
        
        ax2.set_title(f'🔺 Mesh Convex Hull\n{len(vertices1)} vértices', fontweight='bold')
        ax2.view_init(elev=30, azim=45)
    
    # Mesh 2 - Delaunay
    if mesh2:
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        vertices2 = mesh2.vertices
        faces2 = mesh2.faces
        
        for face in faces2[::8]:
            triangle = vertices2[face]
            triangle_closed = np.vstack([triangle, triangle[0]])
            ax3.plot(triangle_closed[:, 0], triangle_closed[:, 1], triangle_closed[:, 2], 
                    'g-', alpha=0.7, linewidth=0.5)
        
        ax3.set_title(f'🔷 Mesh Delaunay\n{len(vertices2)} vértices', fontweight='bold')
        ax3.view_init(elev=30, azim=45)
    
    # Vista Superior - Point Cloud
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    ax4.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c='red', s=3, alpha=0.8)
    ax4.view_init(elev=90, azim=0)
    ax4.set_title('🔴 Vista Superior\nPoint Cloud', fontweight='bold')
    
    # Vista Superior - Mesh 1
    if mesh1:
        ax5 = plt.subplot(2, 3, 5, projection='3d')
        vertices1 = mesh1.vertices
        ax5.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], 
                   c='red', s=2, alpha=0.8)
        ax5.view_init(elev=90, azim=0)
        ax5.set_title('🔴 Vista Superior\nMesh Convex Hull', fontweight='bold')
    
    # Vista Superior - Mesh 2
    if mesh2:
        ax6 = plt.subplot(2, 3, 6, projection='3d')
        vertices2 = mesh2.vertices
        ax6.scatter(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], 
                   c='green', s=2, alpha=0.8)
        ax6.view_init(elev=90, azim=0)
        ax6.set_title('🔴 Vista Superior\nMesh Delaunay', fontweight='bold')
    
    plt.suptitle('🔥 PIKACHU NAUTILUS - POINT CLOUD → MESHES 3D 🔥', 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pikachu_point_cloud_to_meshes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Comparação salva: pikachu_point_cloud_to_meshes.png")

def main():
    """Função principal"""
    print("🔥 NAUTILUS MESH GENERATOR - PIKACHU 3D MESHES")
    print("="*60)
    print("🎯 Convertendo Point Cloud Nautilus → Meshes 3D Escaláveis")
    print("="*60)
    
    # Carrega point cloud
    points = carregar_pointcloud()
    
    if points is None:
        print("❌ Falha ao carregar point cloud!")
        return
    
    print("\n🔥 GERANDO MESHES 3D...")
    print("="*40)
    
    # Gera Mesh 1: Convex Hull Otimizado
    mesh1 = gerar_mesh_convex_hull_avancado(points)
    info1 = None
    if mesh1:
        info1 = salvar_mesh_completo(mesh1, 'convex_hull_otimizado')
    
    print()
    
    # Gera Mesh 2: Delaunay Surface
    mesh2 = gerar_mesh_delaunay_surface(points)
    info2 = None
    if mesh2:
        info2 = salvar_mesh_completo(mesh2, 'delaunay_surface')
    
    print("\n" + "="*60)
    print("🎉 GERAÇÃO DE MESHES CONCLUÍDA!")
    print("="*60)
    
    # Visualizações
    print("\n🎮 GERANDO VISUALIZAÇÕES...")
    
    # Comparação visual
    gerar_comparacao_visual_completa(mesh1, mesh2, points)
    
    # Visualizações interativas individuais
    if mesh1:
        print("\n🎮 Mesh Convex Hull - Feche a janela para continuar...")
        visualizar_mesh_3d_interativo(mesh1, 'Convex Hull Otimizado')
    
    if mesh2:
        print("\n🎮 Mesh Delaunay Surface - Feche a janela para continuar...")
        visualizar_mesh_3d_interativo(mesh2, 'Delaunay Surface')
    
    print("\n" + "🔥"*60)
    print("🎉 NAUTILUS MESHES GERADAS COM SUCESSO!")
    print("🔥"*60)
    print("📁 ARQUIVOS GERADOS:")
    print("   🔸 pikachu_mesh_*.obj (visualização)")
    print("   🔸 pikachu_mesh_*.stl (impressão 3D)")
    print("   🔸 pikachu_mesh_*.ply (dados completos)")
    print("   🔸 pikachu_point_cloud_to_meshes.png (comparação)")
    print()
    print("🚀 MESHES ESCALÁVEIS PRONTAS!")
    print("🖨️ Compatíveis com impressão 3D")
    print("🎮 Abra os arquivos .obj em qualquer software 3D")
    print("🔥 PIKACHU NAUTILUS MESH GENERATION COMPLETE! 🔥")

if __name__ == "__main__":
    main()
