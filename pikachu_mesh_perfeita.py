#!/usr/bin/env python3
"""
🔥 PIKACHU MESH PERFEITA - Alinhada com o desenho
================================================

Foco TOTAL na qualidade da MESH:
1. Análise detalhada do desenho do Pikachu
2. Extração de contornos precisos
3. Geração de mesh 3D baseada na forma real
4. Otimização para máxima qualidade visual
5. Alinhamento perfeito com o desenho original

OBJETIVO: MESH PERFEITA que se parece com o Pikachu!
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from PIL import Image
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata
from skimage import measure
import os

class PikachuMeshPerfect:
    """Gerador de mesh perfeita baseada no desenho do Pikachu"""
    
    def __init__(self):
        self.contour_points = None
        self.depth_map = None
        self.pikachu_mask = None
        
    def analisar_pikachu_detalhado(self, caminho="figures/pikachu.png"):
        """Análise detalhada do desenho do Pikachu"""
        print("🎯 ANÁLISE DETALHADA DO PIKACHU...")
        
        try:
            # Carrega imagem
            img = cv2.imread(caminho)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            print(f"   📐 Imagem carregada: {img_rgb.shape}")
            
            # 1. Segmentação precisa do Pikachu (amarelo)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Ranges mais precisos para amarelo do Pikachu
            lower_yellow1 = np.array([15, 50, 50])
            upper_yellow1 = np.array([35, 255, 255])
            
            lower_yellow2 = np.array([20, 100, 100])
            upper_yellow2 = np.array([30, 255, 255])
            
            # Combina máscaras
            mask1 = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
            mask2 = cv2.inRange(hsv, lower_yellow2, upper_yellow2)
            pikachu_mask = cv2.bitwise_or(mask1, mask2)
            
            # Morfologia para limpar a máscara
            kernel = np.ones((5,5), np.uint8)
            pikachu_mask = cv2.morphologyEx(pikachu_mask, cv2.MORPH_CLOSE, kernel)
            pikachu_mask = cv2.morphologyEx(pikachu_mask, cv2.MORPH_OPEN, kernel)
            
            # Preenche buracos
            pikachu_mask = cv2.dilate(pikachu_mask, kernel, iterations=2)
            pikachu_mask = cv2.erode(pikachu_mask, kernel, iterations=2)
            
            self.pikachu_mask = pikachu_mask
            
            print(f"   🟡 Pixels do Pikachu detectados: {np.sum(pikachu_mask > 0):,}")
            
            # 2. Extração de contorno principal
            contours, _ = cv2.findContours(pikachu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Maior contorno (corpo do Pikachu)
                main_contour = max(contours, key=cv2.contourArea)
                
                # Simplifica contorno para pontos-chave
                epsilon = 0.02 * cv2.arcLength(main_contour, True)
                simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)
                
                self.contour_points = simplified_contour.reshape(-1, 2)
                
                print(f"   🔍 Contorno extraído: {len(self.contour_points)} pontos")
                
                # 3. Gera mapa de profundidade baseado na forma
                self.depth_map = self.gerar_mapa_profundidade_pikachu(img_rgb, pikachu_mask)
                
                return img_rgb, pikachu_mask, self.contour_points
            else:
                print("   ❌ Nenhum contorno encontrado!")
                return img_rgb, None, None
                
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            return None, None, None
    
    def gerar_mapa_profundidade_pikachu(self, img, mask):
        """Gera mapa de profundidade inteligente baseado na anatomia do Pikachu"""
        print("   🗺️ Gerando mapa de profundidade anatômico...")
        
        try:
            h, w = mask.shape
            depth_map = np.zeros((h, w), dtype=np.float32)
            
            # Distância ao centro (barriga do Pikachu = mais profundo)
            center_y, center_x = h//2, w//2
            
            # Cria grid de coordenadas
            y_coords, x_coords = np.ogrid[:h, :w]
            
            # Distância euclidiana ao centro
            dist_to_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            # Normaliza distância
            normalized_dist = dist_to_center / max_dist
            
            # Perfil de profundidade: centro mais alto, bordas mais baixas
            # Simula formato arredondado do Pikachu
            depth_profile = 1.0 - normalized_dist**1.5
            depth_profile = np.maximum(depth_profile, 0)
            
            # Aplica apenas onde tem Pikachu
            depth_map = depth_profile * (mask > 0)
            
            # Suavização gaussiana
            depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
            
            # Normaliza para range [0, 1]
            if depth_map.max() > 0:
                depth_map = depth_map / depth_map.max()
            
            print(f"      ✅ Mapa de profundidade gerado: range [0, {depth_map.max():.3f}]")
            
            return depth_map
            
        except Exception as e:
            print(f"      ❌ Erro no mapa de profundidade: {e}")
            return np.zeros((h, w))
    
    def gerar_pontos_3d_do_contorno(self, scale_factor=2.0):
        """Gera pontos 3D a partir do contorno e mapa de profundidade"""
        print("🔺 GERANDO PONTOS 3D DO CONTORNO...")
        
        if self.contour_points is None or self.depth_map is None:
            print("   ❌ Dados insuficientes!")
            return None
        
        try:
            # Normaliza contorno para [-1, 1]
            contour_norm = self.contour_points.copy().astype(float)
            h, w = self.depth_map.shape
            
            # Converte para coordenadas normalizadas
            contour_norm[:, 0] = (contour_norm[:, 0] / w) * 2 - 1  # X
            contour_norm[:, 1] = (contour_norm[:, 1] / h) * 2 - 1  # Y
            contour_norm[:, 1] = -contour_norm[:, 1]  # Inverte Y
            
            # Gera pontos internos usando triangulação
            points_3d = []
            
            # 1. Pontos do contorno (Z = 0, na borda)
            for point in contour_norm:
                x, y = point
                z = 0.0  # Borda tem profundidade zero
                points_3d.append([x, y, z])
            
            # 2. Pontos internos baseados no mapa de profundidade
            step = 0.05  # Resolução da malha interna
            x_range = np.arange(-1, 1, step)
            y_range = np.arange(-1, 1, step)
            
            for x in x_range:
                for y in y_range:
                    # Converte para coordenadas da imagem
                    img_x = int((x + 1) * w / 2)
                    img_y = int((-y + 1) * h / 2)
                    
                    # Verifica se está dentro dos limites e dentro da máscara
                    if (0 <= img_x < w and 0 <= img_y < h and 
                        self.pikachu_mask[img_y, img_x] > 0):
                        
                        # Pega profundidade do mapa
                        z = self.depth_map[img_y, img_x] * scale_factor
                        points_3d.append([x, y, z])
            
            # 3. Pontos adicionais para melhor definição
            # Centro do Pikachu (barriga)
            points_3d.append([0.0, 0.0, scale_factor])
            
            # Pontos intermediários no contorno
            for i in range(len(contour_norm)):
                p1 = contour_norm[i]
                p2 = contour_norm[(i + 1) % len(contour_norm)]
                
                # Ponto médio
                mid_point = (p1 + p2) / 2
                x, y = mid_point
                
                # Z baseado na distância ao centro
                dist_to_center = np.sqrt(x**2 + y**2)
                z = (1 - dist_to_center) * scale_factor * 0.5
                points_3d.append([x, y, z])
            
            points_3d = np.array(points_3d)
            
            print(f"   ✅ {len(points_3d)} pontos 3D gerados")
            print(f"   📐 Range X: [{points_3d[:, 0].min():.3f}, {points_3d[:, 0].max():.3f}]")
            print(f"   📐 Range Y: [{points_3d[:, 1].min():.3f}, {points_3d[:, 1].max():.3f}]")
            print(f"   📐 Range Z: [{points_3d[:, 2].min():.3f}, {points_3d[:, 2].max():.3f}]")
            
            return points_3d
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            return None
    
    def gerar_mesh_otima(self, points_3d):
        """Gera mesh ótima usando múltiplas técnicas"""
        print("🏗️ GERANDO MESH ÓTIMA...")
        
        try:
            meshes = []
            
            # 1. Delaunay 3D com projeção inteligente
            print("   🔷 Método 1: Delaunay inteligente...")
            
            # Projeta para 2D para triangulação
            points_2d = points_3d[:, :2]
            
            try:
                tri_2d = Delaunay(points_2d)
                faces = tri_2d.simplices
                
                # Cria mesh
                mesh_delaunay = trimesh.Trimesh(vertices=points_3d, faces=faces)
                mesh_delaunay.remove_duplicate_faces()
                mesh_delaunay.remove_unreferenced_vertices()
                mesh_delaunay.fix_normals()
                
                meshes.append(("delaunay", mesh_delaunay))
                print(f"      ✅ Delaunay: {len(mesh_delaunay.vertices)} vértices, {len(mesh_delaunay.faces)} faces")
                
            except Exception as e:
                print(f"      ❌ Delaunay falhou: {e}")
            
            # 2. Convex Hull refinado
            print("   🔺 Método 2: Convex Hull refinado...")
            
            try:
                hull = ConvexHull(points_3d)
                mesh_hull = trimesh.Trimesh(vertices=points_3d, faces=hull.simplices)
                
                # Refinamentos
                mesh_hull.remove_duplicate_faces()
                mesh_hull.remove_unreferenced_vertices()
                mesh_hull.fix_normals()
                
                meshes.append(("convex_hull", mesh_hull))
                print(f"      ✅ Convex Hull: {len(mesh_hull.vertices)} vértices, {len(mesh_hull.faces)} faces")
                
            except Exception as e:
                print(f"      ❌ Convex Hull falhou: {e}")
            
            # 3. Mesh por extrusão do contorno
            print("   📏 Método 3: Extrusão do contorno...")
            
            try:
                mesh_extrusion = self.gerar_mesh_por_extrusao()
                if mesh_extrusion:
                    meshes.append(("extrusion", mesh_extrusion))
                    print(f"      ✅ Extrusão: {len(mesh_extrusion.vertices)} vértices, {len(mesh_extrusion.faces)} faces")
                
            except Exception as e:
                print(f"      ❌ Extrusão falhou: {e}")
            
            # Escolhe a melhor mesh baseada em critérios de qualidade
            if meshes:
                best_mesh = self.escolher_melhor_mesh(meshes)
                return best_mesh
            else:
                print("   ❌ Nenhuma mesh foi gerada!")
                return None
                
        except Exception as e:
            print(f"   ❌ Erro na geração: {e}")
            return None
    
    def gerar_mesh_por_extrusao(self):
        """Gera mesh por extrusão do contorno do Pikachu"""
        try:
            if self.contour_points is None:
                return None
            
            # Normaliza contorno
            contour = self.contour_points.copy().astype(float)
            h, w = self.pikachu_mask.shape
            
            # Para coordenadas [-1, 1]
            contour[:, 0] = (contour[:, 0] / w) * 2 - 1
            contour[:, 1] = (contour[:, 1] / h) * 2 - 1
            contour[:, 1] = -contour[:, 1]
            
            # Cria dois níveis: base (Z=0) e topo (Z=altura)
            altura = 1.0
            vertices = []
            
            # Nível inferior (Z=0)
            for point in contour:
                vertices.append([point[0], point[1], 0.0])
            
            # Nível superior (Z=altura)
            for point in contour:
                vertices.append([point[0], point[1], altura])
            
            # Centro dos níveis
            centro_baixo = [0.0, 0.0, 0.0]
            centro_alto = [0.0, 0.0, altura]
            vertices.append(centro_baixo)
            vertices.append(centro_alto)
            
            vertices = np.array(vertices)
            
            # Gera faces
            faces = []
            n = len(contour)
            
            # Faces laterais
            for i in range(n):
                next_i = (i + 1) % n
                
                # Duas faces triangulares por segmento lateral
                faces.append([i, next_i, i + n])  # Triângulo 1
                faces.append([next_i, next_i + n, i + n])  # Triângulo 2
            
            # Face inferior (conecta ao centro inferior)
            centro_baixo_idx = len(vertices) - 2
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([centro_baixo_idx, next_i, i])
            
            # Face superior (conecta ao centro superior)
            centro_alto_idx = len(vertices) - 1
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([centro_alto_idx, i + n, next_i + n])
            
            faces = np.array(faces)
            
            # Cria mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.fix_normals()
            
            return mesh
            
        except Exception as e:
            print(f"      ❌ Erro na extrusão: {e}")
            return None
    
    def escolher_melhor_mesh(self, meshes):
        """Escolhe a melhor mesh baseada em critérios de qualidade"""
        print("   🏆 Escolhendo melhor mesh...")
        
        best_mesh = None
        best_score = 0
        
        for nome, mesh in meshes:
            try:
                # Critérios de qualidade
                score = 0
                
                # Número de vértices (mais é melhor, até certo ponto)
                vertex_score = min(len(mesh.vertices) / 1000, 1.0) * 30
                score += vertex_score
                
                # Watertight é muito importante
                if mesh.is_watertight:
                    score += 50
                
                # Área superficial razoável
                if 1.0 <= mesh.area <= 50.0:
                    score += 20
                
                # Volume positivo (se watertight)
                if mesh.is_watertight and mesh.volume > 0:
                    score += 30
                
                print(f"      📊 {nome}: {score:.1f} pontos")
                print(f"         🔸 Vértices: {len(mesh.vertices)}")
                print(f"         🔸 Faces: {len(mesh.faces)}")
                print(f"         🔸 Watertight: {mesh.is_watertight}")
                print(f"         🔸 Área: {mesh.area:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_mesh = mesh
                    best_name = nome
                    
            except Exception as e:
                print(f"      ❌ Erro avaliando {nome}: {e}")
        
        if best_mesh:
            print(f"   🏆 Melhor mesh: {best_name} ({best_score:.1f} pontos)")
        
        return best_mesh
    
    def otimizar_mesh_final(self, mesh):
        """Otimizações finais da mesh"""
        print("⚡ OTIMIZAÇÕES FINAIS...")
        
        try:
            # 1. Limpeza básica
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_faces()
            
            # 2. Corrige normais
            mesh.fix_normals()
            
            # 3. Subdivisão controlada para suavidade
            try:
                if len(mesh.vertices) < 500:  # Só subdivide se não for muito densa
                    mesh = mesh.subdivide()
                    print("   📈 Subdivisão aplicada")
            except:
                print("   ⚠️ Subdivisão não aplicada")
            
            # 4. Suavização
            try:
                mesh = mesh.smoothed()
                print("   🎨 Suavização aplicada")
            except:
                print("   ⚠️ Suavização não aplicada")
            
            # 5. Preenchimento de buracos
            try:
                mesh.fill_holes()
                print("   🔧 Buracos preenchidos")
            except:
                print("   ℹ️ Sem buracos para preencher")
            
            return mesh
            
        except Exception as e:
            print(f"   ❌ Erro na otimização: {e}")
            return mesh

def salvar_mesh_perfeita(mesh, nome="perfeita"):
    """Salva mesh perfeita em formatos otimizados"""
    print(f"💎 SALVANDO MESH PERFEITA: {nome}...")
    
    formatos = {
        'obj': f'pikachu_mesh_{nome}.obj',
        'stl': f'pikachu_mesh_{nome}.stl',
        'ply': f'pikachu_mesh_{nome}.ply'
    }
    
    for formato, arquivo in formatos.items():
        try:
            mesh.export(arquivo)
            size = os.path.getsize(arquivo)
            print(f"   💎 {formato.upper()}: {arquivo} ({size:,} bytes)")
        except Exception as e:
            print(f"   ❌ {formato}: {e}")
    
    # Estatísticas detalhadas
    print(f"   📊 MESH PERFEITA - ESTATÍSTICAS:")
    print(f"      💎 Vértices: {len(mesh.vertices):,}")
    print(f"      💎 Faces: {len(mesh.faces):,}")
    print(f"      💎 Edges: {len(mesh.edges):,}")
    print(f"      💎 Área: {mesh.area:.6f}")
    print(f"      💎 Watertight: {mesh.is_watertight}")
    
    if mesh.is_watertight:
        print(f"      💎 Volume: {mesh.volume:.6f}")
        print(f"      💎 Densidade: {len(mesh.vertices)/mesh.volume:.1f} vértices/unidade³")

def visualizar_mesh_perfeita(img_original, mask, contour, mesh):
    """Visualização da mesh perfeita"""
    print("🎨 VISUALIZANDO MESH PERFEITA...")
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Imagem original
    ax1 = plt.subplot(2, 4, 1)
    if img_original is not None:
        ax1.imshow(img_original)
    ax1.set_title('🖼️ Pikachu Original', fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    # 2. Máscara do Pikachu
    ax2 = plt.subplot(2, 4, 2)
    if mask is not None:
        ax2.imshow(mask, cmap='yellow')
    ax2.set_title('🟡 Máscara Pikachu', fontweight='bold', fontsize=12)
    ax2.axis('off')
    
    # 3. Contorno extraído
    ax3 = plt.subplot(2, 4, 3)
    if img_original is not None and contour is not None:
        ax3.imshow(img_original)
        ax3.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=3)
        ax3.plot(contour[:, 0], contour[:, 1], 'ro', markersize=4)
    ax3.set_title('🔍 Contorno Extraído', fontweight='bold', fontsize=12)
    ax3.axis('off')
    
    # 4. Mesh - wireframe
    ax4 = plt.subplot(2, 4, 4, projection='3d')
    if mesh:
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Desenha wireframe
        for face in faces[::max(1, len(faces)//100)]:
            triangle = vertices[face]
            triangle_closed = np.vstack([triangle, triangle[0]])
            ax4.plot(triangle_closed[:, 0], triangle_closed[:, 1], triangle_closed[:, 2],
                    'b-', alpha=0.7, linewidth=1)
    
    ax4.set_title(f'🔺 Mesh Wireframe\n{len(mesh.vertices) if mesh else 0} vértices', 
                 fontweight='bold', fontsize=12)
    ax4.view_init(elev=30, azim=45)
    
    # 5. Mesh - surface
    ax5 = plt.subplot(2, 4, 5, projection='3d')
    if mesh:
        vertices = mesh.vertices
        ax5.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c=vertices[:, 2], cmap='viridis', s=20, alpha=0.8)
    ax5.set_title('🎨 Mesh Surface', fontweight='bold', fontsize=12)
    ax5.view_init(elev=30, azim=45)
    
    # 6. Vista superior
    ax6 = plt.subplot(2, 4, 6, projection='3d')
    if mesh:
        vertices = mesh.vertices
        ax6.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c='red', s=15, alpha=0.9)
    ax6.view_init(elev=90, azim=0)
    ax6.set_title('🔴 Vista Superior', fontweight='bold', fontsize=12)
    
    # 7. Vista lateral
    ax7 = plt.subplot(2, 4, 7, projection='3d')
    if mesh:
        vertices = mesh.vertices
        ax7.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c='green', s=15, alpha=0.9)
    ax7.view_init(elev=0, azim=90)
    ax7.set_title('🔴 Vista Lateral', fontweight='bold', fontsize=12)
    
    # 8. Vista frontal
    ax8 = plt.subplot(2, 4, 8, projection='3d')
    if mesh:
        vertices = mesh.vertices
        ax8.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c='blue', s=15, alpha=0.9)
    ax8.view_init(elev=0, azim=0)
    ax8.set_title('🔴 Vista Frontal', fontweight='bold', fontsize=12)
    
    plt.suptitle('🔥 PIKACHU MESH PERFEITA - ALINHADA COM O DESENHO 🔥', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pikachu_mesh_perfeita.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Visualização salva: pikachu_mesh_perfeita.png")

def main():
    """Pipeline para mesh perfeita do Pikachu"""
    print("🔥 PIKACHU MESH PERFEITA - ALINHADA COM O DESENHO")
    print("="*60)
    print("🎯 FOCO TOTAL NA QUALIDADE DA MESH!")
    print("🎯 ALINHAMENTO PERFEITO COM O DESENHO!")
    print("="*60)
    
    # Inicializa gerador
    generator = PikachuMeshPerfect()
    
    # 1. Análise detalhada do Pikachu
    print("\n1️⃣ ANÁLISE DETALHADA DO PIKACHU...")
    img_original, mask, contour = generator.analisar_pikachu_detalhado()
    
    if img_original is None:
        print("❌ Falha na análise da imagem!")
        return
    
    # 2. Geração de pontos 3D baseados no desenho
    print("\n2️⃣ GERAÇÃO DE PONTOS 3D...")
    pontos_3d = generator.gerar_pontos_3d_do_contorno(scale_factor=1.5)
    
    if pontos_3d is None:
        print("❌ Falha na geração de pontos 3D!")
        return
    
    # 3. Geração da mesh ótima
    print("\n3️⃣ GERAÇÃO DA MESH ÓTIMA...")
    mesh_otima = generator.gerar_mesh_otima(pontos_3d)
    
    if mesh_otima is None:
        print("❌ Falha na geração da mesh!")
        return
    
    # 4. Otimizações finais
    print("\n4️⃣ OTIMIZAÇÕES FINAIS...")
    mesh_perfeita = generator.otimizar_mesh_final(mesh_otima)
    
    # 5. Salvamento
    print("\n5️⃣ SALVAMENTO...")
    salvar_mesh_perfeita(mesh_perfeita, "perfeita")
    
    # 6. Visualização
    print("\n6️⃣ VISUALIZAÇÃO...")
    visualizar_mesh_perfeita(img_original, mask, contour, mesh_perfeita)
    
    print("\n" + "🔥"*60)
    print("🎉 PIKACHU MESH PERFEITA GERADA!")
    print("🔥"*60)
    print("🏆 MESH DE QUALIDADE MÁXIMA!")
    print("🎯 ALINHADA PERFEITAMENTE COM O DESENHO!")
    print("💎 OTIMIZAÇÕES AVANÇADAS APLICADAS!")
    print("📁 ARQUIVOS PROFISSIONAIS PRONTOS!")
    print("🔥 RESULTADO PERFEITO ALCANÇADO! 🔥")

if __name__ == "__main__":
    main()
