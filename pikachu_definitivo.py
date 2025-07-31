#!/usr/bin/env python3
"""
⚡ GERADOR AVANÇADO: Pikachu 3D do jeito CERTO!
============================================

Esta é a versão mais avançada que conseguimos fazer sem o modelo oficial.
Usa técnicas de computer vision e geometria 3D para criar um Pikachu melhor.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import cv2
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import gaussian_filter
import trimesh
from skimage import measure, morphology
import warnings
warnings.filterwarnings('ignore')

def criar_pikachu_perfeito():
    """Cria um Pikachu 3D de alta qualidade"""
    print("="*80)
    print("⚡ CRIANDO PIKACHU 3D - VERSÃO DEFINITIVA ⚡")
    print("="*80)
    print()
    
    # PASSO 1: Carrega e analisa a imagem real
    image, depth_map = processar_imagem_avancada()
    
    # PASSO 2: Gera point cloud inteligente
    points, colors = gerar_pointcloud_inteligente(image, depth_map)
    
    # PASSO 3: Cria mesh com algoritmo avançado
    vertices, faces = criar_mesh_avancado(points, colors)
    
    # PASSO 4: Pós-processamento e suavização
    vertices, faces = pos_processar_mesh(vertices, faces)
    
    # PASSO 5: Visualização completa
    visualizar_resultado_final(image, points, colors, vertices, faces)
    
    # PASSO 6: Salva versão final
    salvar_pikachu_final(image, points, colors, vertices, faces)
    
    print("\n🎉 PIKACHU 3D CRIADO COM SUCESSO!")

def processar_imagem_avancada():
    """Processamento avançado da imagem com estimativa de profundidade"""
    print("📷 PROCESSAMENTO AVANÇADO DA IMAGEM...")
    
    try:
        # Carrega imagem real
        image = Image.open("figures/pikachu.png").convert('RGBA')
        img_array = np.array(image)
        
        print(f"   ✅ Imagem carregada: {image.size}")
        
    except Exception as e:
        print(f"   ⚠️ Usando Pikachu simulado: {e}")
        image, img_array = criar_pikachu_hd()
    
    # Converte para RGB
    if img_array.shape[2] == 4:
        # Remove fundo usando alpha
        alpha = img_array[:, :, 3]
        mask = alpha > 128
        rgb_img = img_array[:, :, :3]
    else:
        rgb_img = img_array
        # Cria máscara removendo fundo branco
        white_threshold = 240
        mask = ~np.all(rgb_img > white_threshold, axis=2)
    
    # ESTIMATIVA DE PROFUNDIDADE AVANÇADA
    depth_map = estimar_profundidade_pikachu(rgb_img, mask)
    
    print(f"   ✅ Depth map criado: {depth_map.shape}")
    
    return image, depth_map

def criar_pikachu_hd():
    """Cria um Pikachu HD para teste"""
    print("🎨 Criando Pikachu HD...")
    
    size = 512
    img = np.ones((size, size, 4)) * 255
    img[:, :, 3] = 0  # Fundo transparente
    
    # Usa PIL para desenhar com anti-aliasing
    image = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    center_x, center_y = size // 2, size // 2
    
    # Corpo (oval)
    body_bbox = [center_x-90, center_y-10, center_x+90, center_y+120]
    draw.ellipse(body_bbox, fill=(255, 255, 0, 255))
    
    # Cabeça (círculo)
    head_bbox = [center_x-80, center_y-120, center_x+80, center_y+40]
    draw.ellipse(head_bbox, fill=(255, 255, 0, 255))
    
    # Orelhas
    ear1_points = [(center_x-50, center_y-80), (center_x-80, center_y-140), (center_x-20, center_y-130)]
    ear2_points = [(center_x+50, center_y-80), (center_x+80, center_y-140), (center_x+20, center_y-130)]
    
    draw.polygon(ear1_points, fill=(255, 255, 0, 255))
    draw.polygon(ear2_points, fill=(255, 255, 0, 255))
    
    # Pontas das orelhas (preto)
    draw.polygon([(center_x-70, center_y-130), (center_x-80, center_y-140), (center_x-60, center_y-140)], 
                fill=(0, 0, 0, 255))
    draw.polygon([(center_x+70, center_y-130), (center_x+80, center_y-140), (center_x+60, center_y-140)], 
                fill=(0, 0, 0, 255))
    
    # Olhos
    draw.ellipse([center_x-35, center_y-70, center_x-15, center_y-50], fill=(0, 0, 0, 255))
    draw.ellipse([center_x+15, center_y-70, center_x+35, center_y-50], fill=(0, 0, 0, 255))
    
    # Bochechas
    draw.ellipse([center_x-70, center_y-30, center_x-40, center_y], fill=(255, 100, 100, 255))
    draw.ellipse([center_x+40, center_y-30, center_x+70, center_y], fill=(255, 100, 100, 255))
    
    # Boca
    draw.arc([center_x-10, center_y-20, center_x+10, center_y-5], 0, 180, fill=(0, 0, 0, 255), width=3)
    
    # Aplica blur suave para anti-aliasing
    image = image.filter(ImageFilter.GaussianBlur(0.5))
    
    return image, np.array(image)

def estimar_profundidade_pikachu(rgb_img, mask):
    """Estima mapa de profundidade inteligente para o Pikachu"""
    print("🧠 ESTIMANDO PROFUNDIDADE INTELIGENTE...")
    
    h, w = mask.shape
    depth_map = np.zeros((h, w), dtype=np.float32)
    
    # MÉTODO 1: Distância do centro (efeito "inchado")
    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # Distância euclidiana do centro
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    normalized_dist = dist_from_center / max_dist
    
    # Profundidade baseada na distância (centro mais "alto")
    depth_from_center = (1 - normalized_dist) * 0.5
    
    # MÉTODO 2: Análise de cores para detalhes
    if rgb_img.shape[2] >= 3:
        # Converte para HSV
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        
        # Amarelo (corpo principal) - mais profundo
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        
        # Vermelho (bochechas) - ligeiramente saliente
        red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        
        # Preto (detalhes) - mais profundo
        black_mask = cv2.inRange(rgb_img, np.array([0, 0, 0]), np.array([50, 50, 50]))
        
        # Aplica profundidades diferentes
        depth_map[yellow_mask > 0] = depth_from_center[yellow_mask > 0] * 0.8
        depth_map[red_mask > 0] = depth_from_center[red_mask > 0] * 1.2  # Bochechas salientes
        depth_map[black_mask > 0] = depth_from_center[black_mask > 0] * 0.3  # Detalhes mais fundos
    
    # MÉTODO 3: Erosão para criar gradiente interno
    for i in range(5):
        kernel = np.ones((5,5), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=i+1)
        depth_addition = (eroded > 0) * (0.1 * (i + 1))
        depth_map += depth_addition
    
    # Aplica máscara final
    depth_map[~mask] = 0
    
    # Suaviza o mapa de profundidade
    depth_map = gaussian_filter(depth_map, sigma=2)
    
    print(f"   ✅ Profundidade: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    
    return depth_map

def gerar_pointcloud_inteligente(image, depth_map):
    """Gera point cloud inteligente usando o mapa de profundidade"""
    print("☁️ GERANDO POINT CLOUD INTELIGENTE...")
    
    img_array = np.array(image)
    h, w = depth_map.shape
    
    points = []
    colors = []
    
    # Parâmetros de densidade
    step = 4  # Densidade alta
    depth_scale = 0.8  # Escala da profundidade
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            if depth_map[y, x] > 0:  # Pixel válido
                # Coordenadas normalizadas
                x_norm = (x / w) * 2 - 1
                y_norm = -(y / h) * 2 + 1  # Inverte Y
                z_norm = depth_map[y, x] * depth_scale
                
                # Cor do pixel
                if img_array.shape[2] >= 3:
                    pixel_color = img_array[y, x, :3] / 255.0
                    if len(pixel_color) == 4:  # RGBA
                        pixel_color = pixel_color[:3]
                else:
                    pixel_color = [1.0, 1.0, 0.0]  # Amarelo padrão
                
                points.append([x_norm, y_norm, z_norm])
                colors.append(pixel_color)
                
                # Adiciona pontos internos para dar volume
                if depth_map[y, x] > 0.3:  # Áreas mais profundas
                    for z_offset in np.linspace(-z_norm * 0.5, 0, 3):
                        points.append([x_norm, y_norm, z_offset])
                        colors.append(pixel_color * 0.9)  # Cor ligeiramente mais escura
    
    points = np.array(points)
    colors = np.array(colors)
    
    print(f"   ✅ Point cloud: {len(points)} pontos")
    print(f"   📊 Range XYZ: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"                 Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"                 Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    return points, colors

def criar_mesh_avancado(points, colors):
    """Cria mesh usando algoritmos avançados"""
    print("🌊 CRIANDO MESH AVANÇADO...")
    
    if len(points) < 4:
        print("   ❌ Pontos insuficientes")
        return np.array([[0,0,0], [1,0,0], [0,1,0]]), np.array([[0,1,2]])
    
    # Remove duplicatas
    unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)
    
    print(f"   🔄 Pontos únicos: {len(unique_points)}")
    
    try:
        # MÉTODO: Triangulação 3D usando ConvexHull
        hull = ConvexHull(unique_points)
        vertices = hull.points
        faces = hull.simplices
        
        print(f"   ✅ ConvexHull: {len(vertices)} vértices, {len(faces)} faces")
        
        # Refina o mesh
        vertices, faces = refinar_mesh(vertices, faces)
        
        return vertices, faces
        
    except Exception as e:
        print(f"   ⚠️ ConvexHull falhou: {e}")
        
        # Fallback: Triangulação por camadas
        return triangular_por_camadas(unique_points)

def refinar_mesh(vertices, faces):
    """Refina o mesh para melhor qualidade"""
    print("   🔧 Refinando mesh...")
    
    try:
        # Cria mesh com trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Remove faces degeneradas
        mesh.remove_degenerate_faces()
        
        # Remove vértices duplicados
        mesh.remove_duplicate_faces()
        
        # Suavização
        mesh = mesh.smoothed()
        
        print(f"   ✅ Mesh refinado: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")
        
        return mesh.vertices, mesh.faces
        
    except Exception as e:
        print(f"   ⚠️ Refinamento falhou: {e}")
        return vertices, faces

def triangular_por_camadas(points):
    """Triangulação alternativa por camadas Z"""
    print("   🥞 Triangulação por camadas...")
    
    # Separa em camadas Z
    z_values = np.unique(points[:, 2])
    z_values = np.sort(z_values)
    
    all_vertices = []
    all_faces = []
    
    for i, z in enumerate(z_values):
        layer_points = points[points[:, 2] == z]
        
        if len(layer_points) > 2:
            base_idx = len(all_vertices)
            all_vertices.extend(layer_points)
            
            # Triangulação 2D da camada
            try:
                points_2d = layer_points[:, :2]
                hull_2d = ConvexHull(points_2d)
                
                for simplex in hull_2d.simplices:
                    face = [base_idx + simplex[0], base_idx + simplex[1], base_idx + simplex[2]]
                    all_faces.append(face)
                    
            except Exception as e:
                print(f"     ⚠️ Erro na camada {i}: {e}")
    
    return np.array(all_vertices), np.array(all_faces)

def pos_processar_mesh(vertices, faces):
    """Pós-processamento final do mesh"""
    print("⚙️ PÓS-PROCESSAMENTO...")
    
    try:
        # Cria mesh trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Operações de limpeza
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Suavização adicional
        mesh = mesh.smoothed()
        
        # Verifica integridade
        print(f"   📊 Mesh final:")
        print(f"      Vértices: {len(mesh.vertices)}")
        print(f"      Faces: {len(mesh.faces)}")
        print(f"      Watertight: {mesh.is_watertight}")
        print(f"      Volume: {mesh.volume:.4f}")
        print(f"      Área: {mesh.area:.4f}")
        
        return mesh.vertices, mesh.faces
        
    except Exception as e:
        print(f"   ⚠️ Erro no pós-processamento: {e}")
        return vertices, faces

def visualizar_resultado_final(image, points, colors, vertices, faces):
    """Visualização completa do resultado final"""
    print("📊 CRIANDO VISUALIZAÇÃO FINAL...")
    
    fig = plt.figure(figsize=(24, 16))
    
    # Layout: 3x4 grid
    
    # 1. Imagem original (grande)
    ax1 = plt.subplot(3, 4, (1, 2))
    ax1.imshow(image)
    ax1.set_title('⚡ PIKACHU ORIGINAL ⚡', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # 2. Point cloud - frontal
    ax2 = plt.subplot(3, 4, 3, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=1, alpha=0.8)
    ax2.set_title('☁️ Point Cloud\n(Frontal)', fontweight='bold')
    ax2.view_init(elev=0, azim=0)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 3. Point cloud - lateral
    ax3 = plt.subplot(3, 4, 4, projection='3d')
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=1, alpha=0.8)
    ax3.set_title('☁️ Point Cloud\n(Lateral)', fontweight='bold')
    ax3.view_init(elev=0, azim=90)
    
    # 4. Point cloud - perspectiva
    ax4 = plt.subplot(3, 4, 5, projection='3d')
    ax4.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=1, alpha=0.8)
    ax4.set_title('☁️ Point Cloud\n(3D)', fontweight='bold')
    ax4.view_init(elev=20, azim=45)
    
    # 5. Mesh wireframe
    ax5 = plt.subplot(3, 4, 6, projection='3d')
    
    # Plota wireframe otimizado
    edges_drawn = set()
    for face in faces[:min(1000, len(faces))]:
        if all(f < len(vertices) for f in face):
            for i in range(3):
                j = (i + 1) % 3
                edge = tuple(sorted([face[i], face[j]]))
                
                if edge not in edges_drawn:
                    edges_drawn.add(edge)
                    v1, v2 = vertices[edge[0]], vertices[edge[1]]
                    ax5.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                            'b-', alpha=0.7, linewidth=0.5)
    
    ax5.set_title('🌊 Mesh Wireframe', fontweight='bold')
    ax5.view_init(elev=20, azim=45)
    
    # 6. Mesh sólido - frontal
    ax6 = plt.subplot(3, 4, 7, projection='3d')
    for face in faces[:min(300, len(faces))]:
        if all(f < len(vertices) for f in face):
            triangle = vertices[face]
            ax6.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                           alpha=0.8, color='gold', edgecolor='orange', linewidth=0.1)
    
    ax6.set_title('🔺 Mesh Sólido\n(Frontal)', fontweight='bold')
    ax6.view_init(elev=0, azim=0)
    
    # 7. Mesh sólido - lateral
    ax7 = plt.subplot(3, 4, 8, projection='3d')
    for face in faces[:min(300, len(faces))]:
        if all(f < len(vertices) for f in face):
            triangle = vertices[face]
            ax7.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                           alpha=0.8, color='yellow', edgecolor='darkorange', linewidth=0.1)
    
    ax7.set_title('🔺 Mesh Sólido\n(Lateral)', fontweight='bold')
    ax7.view_init(elev=0, azim=90)
    
    # 8. Mesh sólido - perspectiva (RESULTADO FINAL)
    ax8 = plt.subplot(3, 4, (9, 12), projection='3d')
    for face in faces[:min(500, len(faces))]:
        if all(f < len(vertices) for f in face):
            triangle = vertices[face]
            ax8.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                           alpha=0.9, color='gold', edgecolor='darkorange', linewidth=0.2)
    
    ax8.set_title('🏆 PIKACHU 3D FINAL 🏆', fontsize=18, fontweight='bold', pad=20)
    ax8.view_init(elev=15, azim=30)
    
    # Remove eixos do resultado final para melhor visualização
    ax8.set_xlabel('')
    ax8.set_ylabel('')
    ax8.set_zlabel('')
    ax8.grid(False)
    
    plt.tight_layout()
    plt.savefig('pikachu_final_definitivo.png', dpi=300, bbox_inches='tight')
    plt.show()

def salvar_pikachu_final(image, points, colors, vertices, faces):
    """Salva o Pikachu final em todos os formatos"""
    print("💾 SALVANDO PIKACHU FINAL...")
    
    timestamp = "final"
    
    # 1. Imagem processada
    image.save(f'pikachu_{timestamp}.png')
    print(f"   ✅ pikachu_{timestamp}.png")
    
    # 2. Point cloud
    np.save(f'pikachu_pointcloud_{timestamp}.npy', points)
    np.save(f'pikachu_colors_{timestamp}.npy', colors)
    
    # Point cloud PLY
    try:
        pc = trimesh.points.PointCloud(points, colors=colors)
        pc.export(f'pikachu_pointcloud_{timestamp}.ply')
        print(f"   ✅ pikachu_pointcloud_{timestamp}.ply")
    except Exception as e:
        print(f"   ⚠️ Erro PLY: {e}")
    
    # 3. Mesh em múltiplos formatos
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # OBJ (para modelagem)
        mesh.export(f'pikachu_mesh_{timestamp}.obj')
        print(f"   ✅ pikachu_mesh_{timestamp}.obj")
        
        # STL (para impressão 3D)
        mesh.export(f'pikachu_mesh_{timestamp}.stl')
        print(f"   ✅ pikachu_mesh_{timestamp}.stl")
        
        # PLY (com cores se possível)
        try:
            # Tenta adicionar cores aos vértices
            if len(colors) >= len(vertices):
                vertex_colors = colors[:len(vertices)]
                mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
            mesh.export(f'pikachu_mesh_{timestamp}.ply')
            print(f"   ✅ pikachu_mesh_{timestamp}.ply")
        except:
            pass
        
        # Estatísticas finais
        print(f"\n   📊 ESTATÍSTICAS FINAIS:")
        print(f"      🔺 Vértices: {len(mesh.vertices)}")
        print(f"      🔺 Faces: {len(mesh.faces)}")
        print(f"      💧 Watertight: {'✅' if mesh.is_watertight else '❌'}")
        print(f"      📏 Volume: {mesh.volume:.6f}")
        print(f"      📐 Área superficial: {mesh.area:.6f}")
        print(f"      📦 Bounding box: {mesh.bounds}")
        
    except Exception as e:
        print(f"   ⚠️ Erro mesh: {e}")
        np.save(f'pikachu_vertices_{timestamp}.npy', vertices)
        np.save(f'pikachu_faces_{timestamp}.npy', faces)
    
    # 4. Visualização
    print(f"   ✅ pikachu_final_definitivo.png")
    
    print(f"\n🎯 TODOS OS ARQUIVOS SALVOS COM SUCESSO!")
    print(f"   Prefixo: pikachu_{timestamp}.*")

if __name__ == "__main__":
    criar_pikachu_perfeito()
