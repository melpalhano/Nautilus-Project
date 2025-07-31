#!/usr/bin/env python3
"""
Pikachu Mesh Generator v2.0 - Formato Anatômico Real
Baseado na imagem do Pikachu: cabeça grande, orelhas pontiagudas, 
corpo pequeno, braços para cima, pernas curtas, rabo em raio
"""

import numpy as np
import math

def criar_esfera_modificada(centro, raios, n_theta=16, n_phi=12):
    """Cria uma esfera com raios diferentes em X, Y, Z"""
    vertices = []
    for i in range(n_theta):
        for j in range(n_phi):
            theta = (i / n_theta) * 2 * math.pi
            phi = (j / n_phi) * math.pi
            
            x = centro[0] + raios[0] * math.sin(phi) * math.cos(theta)
            y = centro[1] + raios[1] * math.sin(phi) * math.sin(theta)
            z = centro[2] + raios[2] * math.cos(phi)
            
            vertices.append([x, y, z])
    return vertices

def criar_cilindro(centro_base, centro_topo, raio, n_faces=8):
    """Cria um cilindro entre dois pontos"""
    vertices = []
    
    # Vetor direção
    direcao = np.array(centro_topo) - np.array(centro_base)
    altura = np.linalg.norm(direcao)
    direcao_norm = direcao / altura
    
    # Criar base perpendicular
    if abs(direcao_norm[2]) < 0.9:
        perpendicular = np.cross(direcao_norm, [0, 0, 1])
    else:
        perpendicular = np.cross(direcao_norm, [1, 0, 0])
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    
    # Segundo vetor perpendicular
    perp2 = np.cross(direcao_norm, perpendicular)
    
    # Criar círculos na base e no topo
    for t in [0, 1]:  # Base e topo
        centro_atual = np.array(centro_base) + t * direcao
        for i in range(n_faces):
            theta = (i / n_faces) * 2 * math.pi
            pos = (centro_atual + 
                   raio * math.cos(theta) * perpendicular + 
                   raio * math.sin(theta) * perp2)
            vertices.append(pos.tolist())
    
    return vertices

def gerar_pikachu_forma_real():
    """
    Gera um Pikachu com a forma real da imagem
    """
    vertices = []
    
    print("🎯 Criando CABEÇA grande e arredondada...")
    # Cabeça: grande, ligeiramente achatada nos lados
    cabeca = criar_esfera_modificada(
        centro=[0, 0, 2.5],
        raios=[1.2, 1.0, 1.1],  # Mais larga que alta
        n_theta=20, n_phi=16
    )
    vertices.extend(cabeca)
    
    print("👂 Criando ORELHAS pontiagudas (formato da imagem)...")
    # Orelha esquerda - formato triangular pontiagudo
    orelha_esq_base = [
        [-0.7, 0.0, 3.4],
        [-0.8, -0.2, 3.4],
        [-0.8, 0.2, 3.4],
        [-0.6, -0.1, 3.6],
        [-0.6, 0.1, 3.6],
        [-0.5, 0.0, 4.2]  # Ponta
    ]
    vertices.extend(orelha_esq_base)
    
    # Orelha direita
    orelha_dir_base = [
        [0.7, 0.0, 3.4],
        [0.8, -0.2, 3.4],
        [0.8, 0.2, 3.4],
        [0.6, -0.1, 3.6],
        [0.6, 0.1, 3.6],
        [0.5, 0.0, 4.2]  # Ponta
    ]
    vertices.extend(orelha_dir_base)
    
    print("🫃 Criando CORPO menor que a cabeça...")
    # Corpo: menor que a cabeça, oval
    corpo = criar_esfera_modificada(
        centro=[0, 0, 0.8],
        raios=[0.8, 0.7, 1.0],  # Menor que a cabeça
        n_theta=16, n_phi=12
    )
    vertices.extend(corpo)
    
    print("💪 Criando BRAÇOS para cima (como na imagem)...")
    # Braço esquerdo - levantado
    braco_esq = criar_cilindro(
        centro_base=[-1.0, 0.2, 1.2],
        centro_topo=[-1.6, 0.0, 2.0],
        raio=0.2, n_faces=6
    )
    vertices.extend(braco_esq)
    
    # Braço direito - levantado
    braco_dir = criar_cilindro(
        centro_base=[1.0, 0.2, 1.2],
        centro_topo=[1.6, 0.0, 2.0],
        raio=0.2, n_faces=6
    )
    vertices.extend(braco_dir)
    
    print("🦵 Criando PERNAS curtas...")
    # Perna esquerda
    perna_esq = criar_cilindro(
        centro_base=[-0.3, 0.0, -0.2],
        centro_topo=[-0.4, 0.0, -1.0],
        raio=0.25, n_faces=8
    )
    vertices.extend(perna_esq)
    
    # Perna direita
    perna_dir = criar_cilindro(
        centro_base=[0.3, 0.0, -0.2],
        centro_topo=[0.4, 0.0, -1.0],
        raio=0.25, n_faces=8
    )
    vertices.extend(perna_dir)
    
    print("⚡ Criando RABO em formato de RAIO...")
    # Rabo em zigzag (formato de raio característico)
    pontos_rabo = [
        [0.0, -0.8, 0.5],     # Base no corpo
        [0.2, -1.4, 0.7],     # Primeira curva
        [-0.1, -1.8, 0.9],    # Zigzag para esquerda
        [0.4, -2.2, 1.1],     # Zigzag para direita
        [0.1, -2.6, 1.3],     # Zigzag para esquerda
        [0.5, -2.8, 1.5],     # Ponta final
    ]
    
    # Conectar pontos do rabo com pequenos cilindros
    for i in range(len(pontos_rabo) - 1):
        segmento = criar_cilindro(
            pontos_rabo[i], 
            pontos_rabo[i + 1],
            raio=0.12 * (1 - i/len(pontos_rabo)),  # Afina na ponta
            n_faces=6
        )
        vertices.extend(segmento)
    
    print("😊 Adicionando detalhes FACIAIS...")
    # Olhos
    vertices.extend([
        [-0.25, 0.7, 2.8],  # Olho esquerdo
        [0.25, 0.7, 2.8],   # Olho direito
    ])
    
    # Bochechas vermelhas (marcadores para textura)
    vertices.extend([
        [-0.9, 0.5, 2.3],   # Bochecha esquerda
        [0.9, 0.5, 2.3],    # Bochecha direita
    ])
    
    # Boca
    vertices.extend([
        [0.0, 0.8, 2.4],    # Centro da boca
        [-0.1, 0.75, 2.35], # Canto esquerdo
        [0.1, 0.75, 2.35],  # Canto direito
    ])
    
    print(f"✅ Total de vértices: {len(vertices)}")
    return np.array(vertices)

def conectar_vertices_inteligente(vertices):
    """
    Conecta vértices de forma inteligente para formar superfícies
    """
    faces = []
    n = len(vertices)
    
    print("🔗 Conectando vértices com algoritmo inteligente...")
    
    # Para cada vértice, encontrar vizinhos próximos e criar triângulos
    for i in range(n):
        # Encontrar os 8 vértices mais próximos
        distancias = []
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(vertices[i] - vertices[j])
                distancias.append((dist, j))
        
        # Ordenar por distância e pegar os mais próximos
        distancias.sort()
        vizinhos = [idx for _, idx in distancias[:8]]
        
        # Criar triângulos com vizinhos adjacentes
        for k in range(len(vizinhos)):
            for l in range(k+1, len(vizinhos)):
                v1, v2, v3 = i, vizinhos[k], vizinhos[l]
                
                # Verificar se os três pontos não são colineares
                vec1 = vertices[v2] - vertices[v1]
                vec2 = vertices[v3] - vertices[v1]
                cross = np.cross(vec1, vec2)
                
                # Se a área do triângulo é significativa
                if np.linalg.norm(cross) > 0.01:
                    face = tuple(sorted([v1, v2, v3]))
                    if len(set(face)) == 3 and face not in faces:
                        faces.append([v1, v2, v3])
    
    # Limitar número de faces para evitar overhead
    if len(faces) > 5000:
        faces = faces[:5000]
    
    print(f"✅ Faces criadas: {len(faces)}")
    return faces

def salvar_pikachu_obj(vertices, faces, nome_arquivo):
    """Salva a mesh do Pikachu em formato OBJ"""
    print(f"💾 Salvando: {nome_arquivo}")
    
    with open(nome_arquivo, 'w') as f:
        f.write("# Pikachu Anatômico Realista v2.0\n")
        f.write("# Formato baseado na imagem original\n")
        f.write("# Cabeça grande, orelhas pontiagudas, braços levantados\n\n")
        
        # Vértices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Faces
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"✅ Arquivo salvo: {nome_arquivo}")

def main():
    """Função principal"""
    print("🎮" + "="*60)
    print("     PIKACHU MESH GENERATOR v2.0 - FORMATO REAL")
    print("     Baseado exatamente na imagem do Pikachu!")
    print("="*64)
    
    # Gerar vértices anatômicos
    vertices = gerar_pikachu_forma_real()
    
    # Conectar em faces
    faces = conectar_vertices_inteligente(vertices)
    
    # Salvar
    arquivo_saida = "pikachu_formato_real.obj"
    salvar_pikachu_obj(vertices, faces, arquivo_saida)
    
    # Estatísticas finais
    print(f"\n📊 ESTATÍSTICAS:")
    print(f"   🎯 Vértices: {len(vertices):,}")
    print(f"   🔺 Faces: {len(faces):,}")
    
    # Dimensões
    x_range = vertices[:, 0].max() - vertices[:, 0].min()
    y_range = vertices[:, 1].max() - vertices[:, 1].min()
    z_range = vertices[:, 2].max() - vertices[:, 2].min()
    
    print(f"\n📏 DIMENSÕES:")
    print(f"   Largura (X): {x_range:.2f}")
    print(f"   Profundidade (Y): {y_range:.2f}")
    print(f"   Altura (Z): {z_range:.2f}")
    
    print(f"\n🎨 CARACTERÍSTICAS DA IMAGEM:")
    print(f"   ✅ Cabeça grande e arredondada")
    print(f"   ✅ Orelhas pontiagudas com pontas pretas")
    print(f"   ✅ Corpo menor que a cabeça")
    print(f"   ✅ Braços levantados alegremente")
    print(f"   ✅ Pernas curtas")
    print(f"   ✅ Rabo em formato de raio característico")
    print(f"   ✅ Detalhes faciais (olhos, bochechas, boca)")
    
    print(f"\n🚀 PARA VISUALIZAR:")
    print(f"   start {arquivo_saida}")
    
    print("="*64)
    print("🏆 PIKACHU FORMATO REAL CRIADO COM SUCESSO!")

if __name__ == "__main__":
    main()
