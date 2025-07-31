#!/usr/bin/env python3
"""
⚡ PROCESSAMENTO REAL: Imagem Pikachu → Point Cloud → Mesh
========================================================

Este script processa a imagem REAL do Pikachu que você forneceu
e demonstra como o Nautilus pode gerar point clouds e meshes.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from pathlib import Path

def processar_pikachu_real():
    """Processa a imagem real do Pikachu"""
    print("="*70)
    print("⚡ PROCESSANDO SUA IMAGEM REAL DO PIKACHU ⚡")
    print("="*70)
    print()
    
    # Carrega a imagem real
    image_path = "figures/pikachu.png"
    print(f"📁 Carregando: {image_path}")
    
    try:
        # Carrega imagem
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        print(f"✅ Imagem carregada com sucesso!")
        print(f"   📐 Dimensões: {image.size}")
        print(f"   🎨 Canais: {img_array.shape}")
        
        # Analisa a imagem
        analisar_imagem_pikachu(image, img_array)
        
        # Simula extração de features
        features = extrair_features_reais(img_array)
        
        # Gera point cloud baseado na imagem real
        points = gerar_pointcloud_real(features, img_array)
        
        # Aplica algoritmo estilo Nautilus
        vertices, faces = nautilus_real_simulation(points)
        
        # Visualiza resultados
        visualizar_resultados_reais(image, points, vertices, faces)
        
        # Salva arquivos
        salvar_arquivos_reais(image, points, vertices, faces)
        
        print("\n🎉 PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

def analisar_imagem_pikachu(image, img_array):
    """Analisa as características da imagem real do Pikachu"""
    print("\n🔍 ANÁLISE DA IMAGEM:")
    
    # Converte para HSV para análise de cores
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Detecta cores dominantes
    cores_unicas = np.unique(img_array.reshape(-1, 3), axis=0)
    print(f"   🎨 Cores únicas detectadas: {len(cores_unicas)}")
    
    # Detecta regiões amarelas (corpo do Pikachu)
    yellow_mask = cv2.inRange(img_hsv, np.array([15, 50, 50]), np.array([35, 255, 255]))
    yellow_pixels = np.sum(yellow_mask > 0)
    
    # Detecta regiões vermelhas (bochechas)
    red_mask = cv2.inRange(img_hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_pixels = np.sum(red_mask > 0)
    
    # Detecta regiões pretas (detalhes)
    black_mask = cv2.inRange(img_array, np.array([0, 0, 0]), np.array([50, 50, 50]))
    black_pixels = np.sum(black_mask > 0)
    
    total_pixels = img_array.shape[0] * img_array.shape[1]
    
    print(f"   🟡 Amarelo: {yellow_pixels/total_pixels*100:.1f}% da imagem")
    print(f"   🔴 Vermelho: {red_pixels/total_pixels*100:.1f}% da imagem")
    print(f"   ⚫ Preto: {black_pixels/total_pixels*100:.1f}% da imagem")
    
    return {
        'yellow_mask': yellow_mask,
        'red_mask': red_mask,
        'black_mask': black_mask,
        'hsv': img_hsv
    }

def extrair_features_reais(img_array):
    """Extrai features reais da imagem usando técnicas de visão computacional"""
    print("\n🧠 EXTRAINDO FEATURES REAIS:")
    
    # Converte para escala de cinza
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detecta bordas
    edges = cv2.Canny(gray, 50, 150)
    
    # Detecta contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Encontra o maior contorno (provavelmente o Pikachu)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        print(f"   🔍 Contornos detectados: {len(contours)}")
        print(f"   📐 Maior contorno - Área: {area:.0f}, Perímetro: {perimeter:.0f}")
        
        # Aproxima contorno para reduzir pontos
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        print(f"   🎯 Pontos do contorno simplificado: {len(approx_contour)}")
        
        return {
            'edges': edges,
            'contours': contours,
            'main_contour': largest_contour,
            'approx_contour': approx_contour,
            'gray': gray
        }
    else:
        print("   ⚠️ Nenhum contorno detectado")
        return {'edges': edges, 'gray': gray}

def gerar_pointcloud_real(features, img_array):
    """Gera point cloud baseado nas features reais da imagem"""
    print("\n☁️ GERANDO POINT CLOUD DA IMAGEM REAL:")
    
    points = []
    
    # Método 1: Pontos baseados em contornos
    if 'main_contour' in features:
        contour = features['main_contour']
        
        # Converte contorno 2D para 3D
        for point in contour:
            x, y = point[0]
            
            # Normaliza coordenadas
            x_norm = (x / img_array.shape[1]) * 2 - 1
            y_norm = (y / img_array.shape[0]) * 2 - 1
            
            # Adiciona múltiplas camadas Z para dar volume
            for z_layer in np.linspace(-0.3, 0.3, 5):
                points.append([x_norm, -y_norm, z_layer])  # Inverte Y
    
    # Método 2: Pontos baseados em pixels coloridos
    h, w = img_array.shape[:2]
    
    # Amostra pontos da imagem
    step = 8  # Pega 1 a cada 8 pixels
    for y in range(0, h, step):
        for x in range(0, w, step):
            pixel = img_array[y, x]
            
            # Pula pixels muito claros (fundo)
            if np.sum(pixel) < 700:  # Não é branco demais
                x_norm = (x / w) * 2 - 1
                y_norm = (y / h) * 2 - 1
                
                # Z baseado na intensidade da cor
                intensity = np.mean(pixel) / 255.0
                z = (intensity - 0.5) * 0.4
                
                points.append([x_norm, -y_norm, z])
    
    points = np.array(points)
    
    # Remove pontos duplicados
    unique_points = np.unique(points, axis=0)
    
    print(f"   ✅ Point cloud gerado: {len(unique_points)} pontos únicos")
    print(f"   📊 Range X: [{unique_points[:, 0].min():.2f}, {unique_points[:, 0].max():.2f}]")
    print(f"   📊 Range Y: [{unique_points[:, 1].min():.2f}, {unique_points[:, 1].max():.2f}]")
    print(f"   📊 Range Z: [{unique_points[:, 2].min():.2f}, {unique_points[:, 2].max():.2f}]")
    
    return unique_points

def nautilus_real_simulation(points):
    """Simula o algoritmo Nautilus real baseado no paper"""
    print("\n🌊 APLICANDO ALGORITMO NAUTILUS:")
    
    print("   🔄 Tokenização estilo Nautilus Shell...")
    
    # Encontra centro dos pontos
    center = np.mean(points, axis=0)
    
    # Calcula distâncias do centro
    distances = np.linalg.norm(points - center, axis=1)
    
    # Ordena pontos em shells concêntricos (como concha nautilus)
    sorted_indices = np.argsort(distances)
    sorted_points = points[sorted_indices]
    
    # Agrupa em shells
    n_shells = 8
    points_per_shell = len(sorted_points) // n_shells
    
    vertices = []
    faces = []
    
    print(f"   🐚 Criando {n_shells} shells concêntricas...")
    
    # Gera faces conectando shells adjacentes
    for shell_idx in range(n_shells - 1):
        start = shell_idx * points_per_shell
        end = (shell_idx + 1) * points_per_shell
        next_end = min((shell_idx + 2) * points_per_shell, len(sorted_points))
        
        current_shell = sorted_points[start:end]
        next_shell = sorted_points[end:next_end]
        
        vertices.extend(current_shell)
        
        # Conecta pontos entre shells com triângulos
        n_connections = min(len(current_shell), len(next_shell)) - 1
        base_idx = len(vertices) - len(current_shell)
        
        for i in range(n_connections):
            if (base_idx + i + 1 < len(vertices) and 
                base_idx + i + len(current_shell) < len(vertices)):
                
                # Triângulo 1
                face1 = [base_idx + i, 
                        base_idx + i + 1, 
                        base_idx + i + len(current_shell)]
                
                # Triângulo 2  
                if base_idx + i + len(current_shell) + 1 < len(vertices):
                    face2 = [base_idx + i + 1,
                            base_idx + i + len(current_shell),
                            base_idx + i + len(current_shell) + 1]
                    
                    faces.extend([face1, face2])
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Limpa faces inválidas
    valid_faces = []
    for face in faces:
        if (len(face) == 3 and 
            all(0 <= f < len(vertices) for f in face) and 
            len(set(face)) == 3):
            valid_faces.append(face)
    
    faces = np.array(valid_faces) if valid_faces else np.array([[0, 1, 2]])
    
    print(f"   ✅ Mesh Nautilus: {len(vertices)} vértices, {len(faces)} faces")
    
    return vertices, faces

def visualizar_resultados_reais(image, points, vertices, faces):
    """Visualiza os resultados do processamento real"""
    print("\n📊 CRIANDO VISUALIZAÇÃO:")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Imagem original (maior)
    ax1 = plt.subplot(3, 2, (1, 2))
    ax1.imshow(image)
    ax1.set_title('⚡ SUA IMAGEM REAL DO PIKACHU ⚡', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Point cloud - vista frontal
    ax2 = plt.subplot(3, 2, 3, projection='3d')
    colors = ['yellow' if p[2] > 0 else 'orange' for p in points]
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=3, alpha=0.8)
    ax2.set_title('☁️ Point Cloud 3D', fontweight='bold')
    ax2.view_init(elev=0, azim=0)
    
    # Point cloud - vista lateral
    ax3 = plt.subplot(3, 2, 4, projection='3d')
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=3, alpha=0.8)
    ax3.set_title('☁️ Point Cloud (Lateral)', fontweight='bold')
    ax3.view_init(elev=0, azim=90)
    
    # Mesh wireframe
    ax4 = plt.subplot(3, 2, 5, projection='3d')
    
    # Plota arestas do mesh
    for face in faces[:min(300, len(faces))]:
        if all(f < len(vertices) for f in face):
            triangle = vertices[face]
            for i in range(3):
                j = (i + 1) % 3
                ax4.plot([triangle[i,0], triangle[j,0]], 
                        [triangle[i,1], triangle[j,1]], 
                        [triangle[i,2], triangle[j,2]], 
                        'b-', alpha=0.6, linewidth=0.8)
    
    ax4.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=4, alpha=0.8)
    ax4.set_title('🌊 Mesh Nautilus', fontweight='bold')
    
    # Mesh sólido
    ax5 = plt.subplot(3, 2, 6, projection='3d')
    
    # Renderiza superfície
    if len(faces) > 0:
        for face in faces[:min(100, len(faces))]:
            if all(f < len(vertices) for f in face):
                triangle = vertices[face]
                ax5.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                               alpha=0.8, color='gold', edgecolor='darkorange')
    
    ax5.set_title('🔺 Resultado Final', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pikachu_processamento_real.png', dpi=200, bbox_inches='tight')
    plt.show()

def salvar_arquivos_reais(image, points, vertices, faces):
    """Salva todos os arquivos gerados"""
    print("\n💾 SALVANDO RESULTADOS:")
    
    # 1. Cópia da imagem original
    image.save('pikachu_original_processado.png')
    print("   ✅ pikachu_original_processado.png")
    
    # 2. Point cloud
    np.save('pikachu_pointcloud_real.npy', points)
    print("   ✅ pikachu_pointcloud_real.npy")
    
    # 3. Point cloud como PLY
    try:
        import trimesh
        pc = trimesh.points.PointCloud(points)
        pc.export('pikachu_pointcloud_real.ply')
        print("   ✅ pikachu_pointcloud_real.ply")
    except:
        print("   ⚠️ PLY não disponível")
    
    # 4. Mesh
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export('pikachu_mesh_nautilus_real.obj')
        print("   ✅ pikachu_mesh_nautilus_real.obj")
        
        # Para impressão 3D
        mesh.export('pikachu_mesh_nautilus_real.stl')
        print("   ✅ pikachu_mesh_nautilus_real.stl")
    except Exception as e:
        print(f"   ⚠️ Erro mesh: {e}")
        np.save('pikachu_mesh_vertices.npy', vertices)
        np.save('pikachu_mesh_faces.npy', faces)
        print("   ✅ pikachu_mesh_vertices.npy + faces.npy")
    
    print("   ✅ pikachu_processamento_real.png")

def mostrar_comando_real():
    """Mostra o comando para usar com o modelo oficial"""
    print("\n" + "="*70)
    print("🚀 COMANDO PARA USAR SEU PIKACHU NO NAUTILUS OFICIAL:")
    print("="*70)
    print()
    print("```bash")
    print("python miche/encode.py \\")
    print("    --config_path miche/shapevae-256.yaml \\") 
    print("    --ckpt_path MODEL_CHECKPOINT.ckpt \\")
    print("    --image_path figures/pikachu.png \\")
    print("    --output_dir ./pikachu_output/")
    print("```")
    print()
    print("📁 Resultado esperado em ./pikachu_output/:")
    print("   🔺 pikachu_mesh.obj (modelo 3D)")
    print("   ☁️  pikachu_pointcloud.ply")
    print("   📊 pikachu_analysis.json")
    print()

if __name__ == "__main__":
    processar_pikachu_real()
    mostrar_comando_real()
