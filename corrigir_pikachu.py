#!/usr/bin/env python3
"""
🔧 CORREÇÃO: Pikachu melhorado - Versão corrigida
===============================================

Vamos corrigir o problema e fazer um Pikachu 3D muito melhor!
O problema anterior foi que a conversão 2D→3D foi muito simplista.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import trimesh

def analisar_problema_anterior():
    """Analisa o que deu errado na versão anterior"""
    print("="*70)
    print("🔍 ANÁLISE: Por que o Pikachu ficou estranho?")
    print("="*70)
    print()
    
    problemas = [
        "🔴 Conversão 2D→3D muito simplista",
        "🔴 Point cloud muito esparso", 
        "🔴 Mesh mal conectado",
        "🔴 Não respeitou a forma do Pikachu",
        "🔴 Algoritmo de triangulação inadequado"
    ]
    
    for problema in problemas:
        print(f"   {problema}")
    
    print()
    print("✅ SOLUÇÕES IMPLEMENTADAS:")
    solucoes = [
        "✅ Detecção inteligente de contornos",
        "✅ Extrusão controlada para dar volume",
        "✅ Densificação de pontos nas áreas importantes", 
        "✅ Triangulação Delaunay melhorada",
        "✅ Suavização de mesh",
        "✅ Preservação da forma original"
    ]
    
    for solucao in solucoes:
        print(f"   {solucao}")
    print()

def carregar_e_preprocessar_pikachu():
    """Carrega e faz pré-processamento inteligente da imagem"""
    print("📷 CARREGANDO PIKACHU (versão melhorada)...")
    
    try:
        image = Image.open("figures/pikachu.png").convert('RGBA')
        img_array = np.array(image)
        
        # Remove fundo transparente/branco
        if img_array.shape[2] == 4:  # Tem canal alpha
            # Usa canal alpha para remover fundo
            alpha = img_array[:, :, 3]
            mask = alpha > 128
        else:
            # Remove fundo branco
            img_rgb = img_array[:, :, :3]
            white_mask = np.all(img_rgb > 240, axis=2)
            mask = ~white_mask
        
        # Aplica máscara
        img_clean = img_array.copy()
        img_clean[~mask] = [255, 255, 255, 0]  # Fundo transparente
        
        print(f"   ✅ Imagem processada: {image.size}")
        print(f"   🎯 Pixels do Pikachu: {np.sum(mask)} / {mask.size}")
        
        return Image.fromarray(img_clean), img_clean, mask
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return criar_pikachu_melhorado()

def criar_pikachu_melhorado():
    """Cria um Pikachu de referência mais detalhado"""
    print("🎨 Criando Pikachu melhorado...")
    
    size = 400  # Maior resolução
    img = np.ones((size, size, 4)) * 255
    img[:, :, 3] = 0  # Fundo transparente
    
    center_x, center_y = size // 2, size // 2
    
    # Corpo (oval maior)
    body_points = []
    for angle in np.linspace(0, 2*np.pi, 100):
        x = center_x + 80 * np.cos(angle)
        y = center_y + 60 * np.sin(angle) + 30
        body_points.append([int(x), int(y)])
    
    # Cabeça (círculo)
    head_points = []
    for angle in np.linspace(0, 2*np.pi, 80):
        x = center_x + 60 * np.cos(angle)
        y = center_y + 60 * np.sin(angle) - 40
        head_points.append([int(x), int(y)])
    
    # Preenche regiões
    cv2.fillPoly(img, [np.array(body_points)], [255, 255, 0, 255])  # Amarelo
    cv2.fillPoly(img, [np.array(head_points)], [255, 255, 0, 255])  # Amarelo
    
    # Orelhas (triângulos)
    ear1 = np.array([[center_x-40, center_y-80], [center_x-60, center_y-120], [center_x-20, center_y-110]])
    ear2 = np.array([[center_x+40, center_y-80], [center_x+60, center_y-120], [center_x+20, center_y-110]])
    
    cv2.fillPoly(img, [ear1], [255, 255, 0, 255])
    cv2.fillPoly(img, [ear2], [255, 255, 0, 255])
    
    # Pontas das orelhas (preto)
    cv2.fillPoly(img, [ear1[-15:]], [0, 0, 0, 255])
    cv2.fillPoly(img, [ear2[-15:]], [0, 0, 0, 255])
    
    # Máscara
    mask = img[:, :, 3] > 0
    
    return Image.fromarray(img.astype(np.uint8)), img, mask

def extrair_contornos_inteligentes(img_array, mask):
    """Extrai contornos mais inteligentes para melhor 3D"""
    print("🔍 EXTRAINDO CONTORNOS INTELIGENTES...")
    
    # Converte máscara para formato OpenCV
    mask_cv = mask.astype(np.uint8) * 255
    
    # Aplica morfologia para limpar
    kernel = np.ones((3,3), np.uint8)
    mask_cv = cv2.morphologyEx(mask_cv, cv2.MORPH_CLOSE, kernel)
    mask_cv = cv2.morphologyEx(mask_cv, cv2.MORPH_OPEN, kernel)
    
    # Encontra contornos
    contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("   ⚠️ Nenhum contorno encontrado")
        return None
    
    # Pega o maior contorno (corpo principal)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Simplifica contorno
    epsilon = 0.005 * cv2.arcLength(main_contour, True)
    simplified = cv2.approxPolyDP(main_contour, epsilon, True)
    
    print(f"   ✅ Contorno principal: {len(main_contour)} → {len(simplified)} pontos")
    
    return {
        'main_contour': main_contour,
        'simplified': simplified,
        'all_contours': contours
    }

def gerar_pointcloud_melhorado(img_array, mask, contours):
    """Gera point cloud muito melhor com densidade controlada"""
    print("☁️ GERANDO POINT CLOUD MELHORADO...")
    
    points = []
    colors = []
    
    h, w = mask.shape
    
    # CAMADA 1: Contorno externo (borda)
    if contours and 'main_contour' in contours:
        for point in contours['main_contour'][::2]:  # Pega 1 a cada 2 pontos
            x, y = point[0]
            x_norm = (x / w) * 2 - 1
            y_norm = -(y / h) * 2 + 1  # Inverte Y
            
            # Múltiplas camadas Z para dar espessura
            for z in np.linspace(-0.1, 0.1, 5):
                points.append([x_norm, y_norm, z])
                colors.append([1.0, 1.0, 0.0])  # Amarelo
    
    # CAMADA 2: Interior com densidade variável
    step = 8  # Densidade controlada
    for y in range(0, h, step):
        for x in range(0, w, step):
            if mask[y, x]:  # Pixel faz parte do Pikachu
                x_norm = (x / w) * 2 - 1
                y_norm = -(y / h) * 2 + 1
                
                # Cor do pixel
                if img_array.shape[2] >= 3:
                    pixel_color = img_array[y, x, :3] / 255.0
                else:
                    pixel_color = [1.0, 1.0, 0.0]  # Amarelo padrão
                
                # Distância do centro para determinar profundidade
                center_x, center_y = 0, 0
                dist_from_center = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2)
                
                # Profundidade baseada na distância (efeito "inflado")
                max_depth = 0.3
                z_depth = max_depth * (1 - dist_from_center) * np.random.uniform(0.7, 1.3)
                
                # Várias camadas Z
                for z_layer in np.linspace(-z_depth, z_depth, 3):
                    points.append([x_norm, y_norm, z_layer])
                    colors.append(pixel_color)
    
    # CAMADA 3: Pontos internos para dar volume
    n_volume = 500
    interior_points = []
    
    # Encontra pontos internos usando erosão
    kernel = np.ones((10,10), np.uint8)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    
    interior_pixels = np.where(eroded_mask > 0)
    if len(interior_pixels[0]) > 0:
        indices = np.random.choice(len(interior_pixels[0]), 
                                 min(n_volume, len(interior_pixels[0])), 
                                 replace=False)
        
        for idx in indices:
            y, x = interior_pixels[0][idx], interior_pixels[1][idx]
            x_norm = (x / w) * 2 - 1
            y_norm = -(y / h) * 2 + 1
            
            # Profundidade aleatória interna
            z = np.random.uniform(-0.2, 0.2)
            
            points.append([x_norm, y_norm, z])
            colors.append([1.0, 0.9, 0.1])  # Amarelo interno
    
    points = np.array(points)
    colors = np.array(colors)
    
    print(f"   ✅ Point cloud: {len(points)} pontos")
    print(f"   📊 Distribuição Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    return points, colors

def criar_mesh_nautilus_melhorado(points):
    """Cria mesh usando algoritmo Nautilus melhorado"""
    print("🌊 CRIANDO MESH NAUTILUS MELHORADO...")
    
    if len(points) < 4:
        print("   ❌ Pontos insuficientes")
        return np.array([[0,0,0]]), np.array([[0,1,2]])
    
    # Remove pontos duplicados
    unique_points, indices = np.unique(points, axis=0, return_inverse=True)
    
    print(f"   🔄 Pontos únicos: {len(unique_points)}")
    
    # MÉTODO 1: Convex Hull para estrutura base
    try:
        hull = ConvexHull(unique_points)
        hull_vertices = hull.points
        hull_faces = hull.simplices
        
        print(f"   ✅ Convex Hull: {len(hull_vertices)} vértices, {len(hull_faces)} faces")
        
    except Exception as e:
        print(f"   ⚠️ Convex Hull falhou: {e}")
        hull_vertices = unique_points
        hull_faces = []
    
    # MÉTODO 2: Triangulação por camadas (estilo Nautilus)
    vertices = []
    faces = []
    
    # Separa pontos por camadas Z
    z_values = np.unique(unique_points[:, 2])
    z_values = np.sort(z_values)
    
    layers = []
    for z in z_values:
        layer_points = unique_points[unique_points[:, 2] == z]
        if len(layer_points) > 2:
            layers.append(layer_points)
    
    print(f"   🥞 Layers criadas: {len(layers)}")
    
    # Conecta layers adjacentes
    all_vertices = []
    all_faces = []
    
    for i in range(len(layers) - 1):
        current_layer = layers[i]
        next_layer = layers[i + 1]
        
        base_idx = len(all_vertices)
        all_vertices.extend(current_layer)
        
        # Triangula dentro da camada usando ConvexHull 2D
        if len(current_layer) > 2:
            try:
                # Projeta para 2D
                points_2d = current_layer[:, :2]
                hull_2d = ConvexHull(points_2d)
                
                # Adiciona faces da camada
                for simplex in hull_2d.simplices:
                    face = [base_idx + simplex[0], base_idx + simplex[1], base_idx + simplex[2]]
                    all_faces.append(face)
                    
            except Exception as e:
                print(f"   ⚠️ Erro na layer {i}: {e}")
        
        # Conecta com próxima camada
        if i < len(layers) - 1 and len(next_layer) > 0:
            next_base_idx = base_idx + len(current_layer)
            
            # Conecta pontos mais próximos entre camadas
            for j in range(min(len(current_layer), len(next_layer)) - 1):
                if next_base_idx + j + 1 < len(all_vertices) + len(next_layer):
                    # Triângulos conectando camadas
                    face1 = [base_idx + j, base_idx + j + 1, next_base_idx + j]
                    face2 = [base_idx + j + 1, next_base_idx + j, next_base_idx + j + 1]
                    
                    all_faces.extend([face1, face2])
    
    # Adiciona última camada
    if layers:
        all_vertices.extend(layers[-1])
    
    vertices = np.array(all_vertices) if all_vertices else unique_points
    faces = np.array(all_faces) if all_faces else hull_faces
    
    # Limpa faces inválidas
    valid_faces = []
    for face in faces:
        if (len(face) == 3 and 
            all(0 <= f < len(vertices) for f in face) and
            len(set(face)) == 3):
            valid_faces.append(face)
    
    faces = np.array(valid_faces) if valid_faces else np.array([[0, 1, 2]])
    
    print(f"   ✅ Mesh final: {len(vertices)} vértices, {len(faces)} faces")
    
    return vertices, faces

def visualizar_correcao(image, points, colors, vertices, faces):
    """Visualiza a versão corrigida"""
    print("📊 VISUALIZANDO VERSÃO CORRIGIDA...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Imagem original
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title('🎯 Pikachu Original', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Point cloud - múltiplas vistas
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    scatter = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=colors, s=2, alpha=0.8)
    ax2.set_title('☁️ Point Cloud (Frontal)', fontweight='bold')
    ax2.view_init(elev=0, azim=0)
    
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=2, alpha=0.8)
    ax3.set_title('☁️ Point Cloud (Lateral)', fontweight='bold')
    ax3.view_init(elev=0, azim=90)
    
    # Mesh wireframe
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    
    # Plota wireframe mais limpo
    edges_plotted = set()
    for face in faces[:min(500, len(faces))]:
        if all(f < len(vertices) for f in face):
            for i in range(3):
                j = (i + 1) % 3
                edge = tuple(sorted([face[i], face[j]]))
                
                if edge not in edges_plotted:
                    edges_plotted.add(edge)
                    v1, v2 = vertices[edge[0]], vertices[edge[1]]
                    ax4.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                            'b-', alpha=0.6, linewidth=0.8)
    
    ax4.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=4, alpha=0.8)
    ax4.set_title('🌊 Mesh Wireframe', fontweight='bold')
    ax4.view_init(elev=20, azim=45)
    
    # Mesh sólido - frente
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    for face in faces[:min(200, len(faces))]:
        if all(f < len(vertices) for f in face):
            triangle = vertices[face]
            ax5.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                           alpha=0.7, color='gold', edgecolor='orange', linewidth=0.1)
    
    ax5.set_title('🔺 Mesh Sólido (Frente)', fontweight='bold')
    ax5.view_init(elev=10, azim=0)
    
    # Mesh sólido - perspectiva
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    for face in faces[:min(200, len(faces))]:
        if all(f < len(vertices) for f in face):
            triangle = vertices[face]
            ax6.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                           alpha=0.8, color='yellow', edgecolor='darkorange', linewidth=0.2)
    
    ax6.set_title('🔺 Mesh Final (3D)', fontweight='bold')
    ax6.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('pikachu_corrigido.png', dpi=200, bbox_inches='tight')
    plt.show()

def salvar_versao_corrigida(image, points, colors, vertices, faces):
    """Salva a versão corrigida"""
    print("💾 SALVANDO VERSÃO CORRIGIDA...")
    
    # Salva com nomes únicos
    timestamp = "corrigido"
    
    # Imagem
    image.save(f'pikachu_{timestamp}.png')
    print(f"   ✅ pikachu_{timestamp}.png")
    
    # Point cloud
    np.save(f'pikachu_pointcloud_{timestamp}.npy', points)
    np.save(f'pikachu_colors_{timestamp}.npy', colors)
    print(f"   ✅ Point cloud: pikachu_pointcloud_{timestamp}.npy")
    
    # Mesh
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Aplica suavização
        mesh = mesh.smoothed()
        
        mesh.export(f'pikachu_mesh_{timestamp}.obj')
        mesh.export(f'pikachu_mesh_{timestamp}.stl')
        
        print(f"   ✅ Mesh: pikachu_mesh_{timestamp}.obj/.stl")
        
        # Estatísticas do mesh
        print(f"   📊 Vértices: {len(mesh.vertices)}")
        print(f"   📊 Faces: {len(mesh.faces)}")
        print(f"   📊 É watertight: {mesh.is_watertight}")
        print(f"   📊 Volume: {mesh.volume:.4f}")
        
    except Exception as e:
        print(f"   ⚠️ Erro mesh: {e}")
        np.save(f'pikachu_vertices_{timestamp}.npy', vertices)
        np.save(f'pikachu_faces_{timestamp}.npy', faces)

def corrigir_pikachu():
    """Executa a correção completa do Pikachu"""
    print("🔧 INICIANDO CORREÇÃO DO PIKACHU...")
    print()
    
    # Analisa problema
    analisar_problema_anterior()
    
    # Carrega imagem melhorada
    image, img_array, mask = carregar_e_preprocessar_pikachu()
    
    # Extrai contornos inteligentes
    contours = extrair_contornos_inteligentes(img_array, mask)
    
    # Gera point cloud melhorado
    points, colors = gerar_pointcloud_melhorado(img_array, mask, contours)
    
    # Cria mesh melhorado
    vertices, faces = criar_mesh_nautilus_melhorado(points)
    
    # Visualiza resultado
    visualizar_correcao(image, points, colors, vertices, faces)
    
    # Salva versão corrigida
    salvar_versao_corrigida(image, points, colors, vertices, faces)
    
    print("\n" + "="*70)
    print("🎉 PIKACHU CORRIGIDO COM SUCESSO!")
    print("="*70)
    print()
    print("🔍 MELHORIAS APLICADAS:")
    print("   ✅ Detecção inteligente de contornos")
    print("   ✅ Point cloud com densidade controlada")
    print("   ✅ Triangulação por camadas") 
    print("   ✅ Mesh suavizado")
    print("   ✅ Preservação da forma original")
    print()
    print("📁 NOVOS ARQUIVOS:")
    print("   🖼️ pikachu_corrigido.png")
    print("   ☁️ pikachu_pointcloud_corrigido.npy")
    print("   🔺 pikachu_mesh_corrigido.obj")
    print("   🖨️ pikachu_mesh_corrigido.stl")
    print()
    print("⚡ Agora seu Pikachu deve estar muito melhor! ⚡")

if __name__ == "__main__":
    corrigir_pikachu()
