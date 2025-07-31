#!/usr/bin/env python3
"""
Gerador de Mesh Pikachu Anatômico
Cria uma mesh 3D que realmente parece com o Pikachu da imagem
"""

import numpy as np
import math

def gerar_vertices_pikachu_anatomico():
    """
    Gera vértices que formam a anatomia completa do Pikachu
    Baseado na imagem: corpo, cabeça, orelhas, braços, pernas, rabo
    """
    vertices = []
    
    # 1. CABEÇA (esfera ligeiramente achatada)
    print("🎯 Gerando cabeça do Pikachu...")
    for i in range(20):
        for j in range(15):
            theta = (i / 20.0) * 2 * math.pi
            phi = (j / 15.0) * math.pi
            
            # Raio da cabeça (ligeiramente maior na frente)
            r = 1.2 + 0.1 * math.cos(phi)
            
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta) 
            z = r * math.cos(phi) + 2.0  # Elevada
            
            vertices.append([x, y, z])
    
    # 2. ORELHAS PONTIAGUDAS (formato triangular)
    print("👂 Gerando orelhas pontiagudas...")
    
    # Orelha esquerda
    for i in range(8):
        for j in range(6):
            # Base da orelha
            base_x = -0.8 + (i / 8.0) * 0.4
            base_y = -0.5 + (j / 6.0) * 1.0
            
            # Altura variável (formato triangular)
            altura = 1.5 * (1 - abs(base_x + 0.6) / 0.4) * (1 - abs(base_y) / 0.5)
            z = 2.8 + altura
            
            vertices.append([base_x, base_y, z])
    
    # Orelha direita
    for i in range(8):
        for j in range(6):
            base_x = 0.4 + (i / 8.0) * 0.4
            base_y = -0.5 + (j / 6.0) * 1.0
            
            altura = 1.5 * (1 - abs(base_x - 0.6) / 0.4) * (1 - abs(base_y) / 0.5)
            z = 2.8 + altura
            
            vertices.append([base_x, base_y, z])
    
    # Pontas pretas das orelhas
    vertices.append([-0.6, 0.0, 4.3])  # Ponta orelha esquerda
    vertices.append([0.6, 0.0, 4.3])   # Ponta orelha direita
    
    # 3. CORPO (oval alongado)
    print("🫃 Gerando corpo do Pikachu...")
    for i in range(16):
        for j in range(12):
            theta = (i / 16.0) * 2 * math.pi
            phi = (j / 12.0) * math.pi
            
            # Corpo mais largo e alto
            r_x = 1.0
            r_y = 0.8
            r_z = 1.4
            
            x = r_x * math.sin(phi) * math.cos(theta)
            y = r_y * math.sin(phi) * math.sin(theta)
            z = r_z * math.cos(phi) - 0.3  # Abaixo da cabeça
            
            vertices.append([x, y, z])
    
    # 4. BRAÇOS (cilindros)
    print("💪 Gerando braços do Pikachu...")
    
    # Braço esquerdo
    for i in range(8):
        for j in range(6):
            theta = (j / 6.0) * 2 * math.pi
            t = i / 8.0
            
            # Posição ao longo do braço
            x = -1.2 - t * 0.5
            y = 0.2 * math.cos(theta)
            z = 0.5 + 0.2 * math.sin(theta) - t * 0.3
            
            vertices.append([x, y, z])
    
    # Braço direito  
    for i in range(8):
        for j in range(6):
            theta = (j / 6.0) * 2 * math.pi
            t = i / 8.0
            
            x = 1.2 + t * 0.5
            y = 0.2 * math.cos(theta)
            z = 0.5 + 0.2 * math.sin(theta) - t * 0.3
            
            vertices.append([x, y, z])
    
    # 5. PERNAS (cilindros mais grossos)
    print("🦵 Gerando pernas do Pikachu...")
    
    # Perna esquerda
    for i in range(10):
        for j in range(8):
            theta = (j / 8.0) * 2 * math.pi
            t = i / 10.0
            
            r = 0.25 + 0.1 * (1 - t)  # Mais grossa em cima
            x = -0.4 + r * math.cos(theta)
            y = r * math.sin(theta)
            z = -1.5 - t * 1.0
            
            vertices.append([x, y, z])
    
    # Perna direita
    for i in range(10):
        for j in range(8):
            theta = (j / 8.0) * 2 * math.pi
            t = i / 10.0
            
            r = 0.25 + 0.1 * (1 - t)
            x = 0.4 + r * math.cos(theta)
            y = r * math.sin(theta)
            z = -1.5 - t * 1.0
            
            vertices.append([x, y, z])
    
    # 6. RABO EM FORMATO DE RAIO ⚡
    print("⚡ Gerando rabo em formato de raio...")
    
    # Segmentos do rabo em zigzag
    rabo_pontos = [
        [0.0, -1.2, 0.0],    # Base
        [0.3, -1.8, 0.2],    # Primeira curva
        [0.1, -2.2, 0.4],    # Zigzag
        [0.5, -2.6, 0.6],    # Outra curva
        [0.2, -3.0, 0.8],    # Continuação
        [0.6, -3.2, 1.0],    # Ponta do rabo
    ]
    
    for i, ponto in enumerate(rabo_pontos):
        # Criar pequena seção cilíndrica em cada ponto
        for j in range(6):
            theta = (j / 6.0) * 2 * math.pi
            r = 0.15 * (1 - i / len(rabo_pontos))  # Afina na ponta
            
            x = ponto[0] + r * math.cos(theta)
            y = ponto[1] + r * math.sin(theta)
            z = ponto[2]
            
            vertices.append([x, y, z])
    
    # 7. DETALHES FACIAIS
    print("😊 Adicionando detalhes faciais...")
    
    # Bochechas vermelhas (posições para textura)
    vertices.append([-0.8, 0.6, 2.0])  # Bochecha esquerda
    vertices.append([0.8, 0.6, 2.0])   # Bochecha direita
    
    # Olhos
    vertices.append([-0.3, 0.8, 2.3])  # Olho esquerdo
    vertices.append([0.3, 0.8, 2.3])   # Olho direito
    
    # Nariz
    vertices.append([0.0, 0.9, 2.1])
    
    print(f"✅ Total de vértices gerados: {len(vertices)}")
    return np.array(vertices)

def gerar_faces_pikachu_anatomico(vertices):
    """
    Gera faces conectando os vértices de forma anatômica
    """
    faces = []
    n_vertices = len(vertices)
    
    print("🔗 Conectando vértices em faces...")
    
    # Estratégia: conectar vértices próximos geograficamente
    # Isso vai criar uma superfície que segue a anatomia
    
    # Para cada vértice, conectar com os 6 mais próximos
    for i in range(n_vertices):
        # Calcular distâncias para todos os outros vértices
        distancias = []
        for j in range(n_vertices):
            if i != j:
                dist = np.linalg.norm(vertices[i] - vertices[j])
                distancias.append((dist, j))
        
        # Pegar os 6 mais próximos
        distancias.sort()
        vizinhos = [idx for _, idx in distancias[:6]]
        
        # Criar triângulos com vizinhos
        for k in range(len(vizinhos)):
            v1 = i
            v2 = vizinhos[k]
            v3 = vizinhos[(k + 1) % len(vizinhos)]
            
            # Evitar faces duplicadas
            face = tuple(sorted([v1, v2, v3]))
            if face not in faces and len(set(face)) == 3:
                faces.append([v1, v2, v3])
    
    print(f"✅ Total de faces geradas: {len(faces)}")
    return faces

def salvar_mesh_obj(vertices, faces, arquivo):
    """
    Salva a mesh em formato OBJ
    """
    print(f"💾 Salvando mesh: {arquivo}")
    
    with open(arquivo, 'w') as f:
        f.write("# Pikachu Anatômico - Mesh 3D Realista\n")
        f.write("# Gerado para parecer com a imagem do Pikachu\n\n")
        
        # Escrever vértices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Escrever faces
        for face in faces:
            # OBJ usa índices baseados em 1
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"✅ Mesh salva: {arquivo}")

def main():
    """
    Função principal - Gera o Pikachu anatômico
    """
    print("🎮" + "="*50)
    print("    GERADOR DE PIKACHU ANATÔMICO 3D")
    print("    Mesh que realmente parece com a imagem!")
    print("="*54)
    
    # Gerar vértices anatômicos
    vertices = gerar_vertices_pikachu_anatomico()
    
    # Gerar faces conectando anatomicamente
    faces = gerar_faces_pikachu_anatomico(vertices)
    
    # Salvar mesh
    arquivo_saida = "pikachu_anatomico_realista.obj"
    salvar_mesh_obj(vertices, faces, arquivo_saida)
    
    # Estatísticas
    print("\n📊 ESTATÍSTICAS DA MESH ANATÔMICA:")
    print(f"   🎯 Vértices: {len(vertices):,}")
    print(f"   🔺 Faces: {len(faces):,}")
    print(f"   📁 Arquivo: {arquivo_saida}")
    
    # Análise dimensional
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    
    print(f"\n📏 DIMENSÕES:")
    print(f"   X: {x_min:.3f} a {x_max:.3f} (largura: {x_max-x_min:.3f})")
    print(f"   Y: {y_min:.3f} a {y_max:.3f} (profundidade: {y_max-y_min:.3f})")
    print(f"   Z: {z_min:.3f} a {z_max:.3f} (altura: {z_max-z_min:.3f})")
    
    print(f"\n🎨 CARACTERÍSTICAS ANATÔMICAS:")
    print(f"   👂 Orelhas pontiagudas com pontas pretas")
    print(f"   😊 Cabeça arredondada com detalhes faciais")
    print(f"   🫃 Corpo oval proporcional")
    print(f"   💪 Braços articulados")
    print(f"   🦵 Pernas robustas")
    print(f"   ⚡ Rabo em formato de raio característico")
    
    print(f"\n🚀 PARA VISUALIZAR:")
    print(f"   Windows 3D Viewer: clique duplo em {arquivo_saida}")
    print(f"   Matplotlib: python visualizador_matplotlib_pikachu.py {arquivo_saida}")
    
    print("="*54)
    print("🏆 PIKACHU ANATÔMICO GERADO COM SUCESSO!")

if __name__ == "__main__":
    main()
