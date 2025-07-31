#!/usr/bin/env python3
"""
PIKACHU MESH SUPREMA - Baseado na imagem exata
Cria mesh 3D com as proporções e poses da imagem fornecida
+ Visualizador matplotlib integrado
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def criar_pikachu_imagem_exata():
    """
    Cria mesh baseada exatamente na imagem do Pikachu
    Análise da imagem: cabeça grande, braços para cima, corpo pequeno, rabo raio
    """
    vertices = []
    
    print("🎯 Analisando imagem do Pikachu...")
    print("   - Cabeça: GRANDE e redonda (proporção dominante)")
    print("   - Orelhas: Pontiagudas, inclinadas para fora")
    print("   - Braços: Levantados em pose alegre")
    print("   - Corpo: Pequeno, formato pêra")
    print("   - Rabo: Formato de raio característico")
    
    # =============================================================================
    # 1. CABEÇA - Elemento dominante da imagem (60% da figura)
    # =============================================================================
    print("\n🎯 Criando CABEÇA (elemento principal)...")
    
    # Cabeça esférica grande - centro da composição
    centro_cabeca = [0, 0, 1.5]
    raio_cabeca = 1.0
    
    # Criar esfera da cabeça com boa resolução
    for i in range(25):  # Alta resolução para suavidade
        for j in range(20):
            u = i / 24.0
            v = j / 19.0
            
            theta = u * 2 * math.pi
            phi = v * math.pi
            
            # Ligeiramente achatada nas laterais (como na imagem)
            x = centro_cabeca[0] + raio_cabeca * 0.95 * math.sin(phi) * math.cos(theta)
            y = centro_cabeca[1] + raio_cabeca * math.sin(phi) * math.sin(theta)
            z = centro_cabeca[2] + raio_cabeca * math.cos(phi)
            
            vertices.append([x, y, z])
    
    # =============================================================================
    # 2. ORELHAS - Formato pontiagudo inclinado da imagem
    # =============================================================================
    print("👂 Criando ORELHAS pontiagudas...")
    
    # Orelha ESQUERDA (vista do Pikachu)
    orelha_esq_pontos = [
        # Base da orelha
        [-0.6, -0.2, 2.2], [-0.7, 0.0, 2.2], [-0.7, 0.2, 2.2],
        [-0.5, -0.15, 2.4], [-0.6, 0.0, 2.4], [-0.6, 0.15, 2.4],
        # Meio da orelha
        [-0.4, -0.1, 2.7], [-0.45, 0.0, 2.7], [-0.45, 0.1, 2.7],
        # Ponta da orelha
        [-0.3, 0.0, 3.0]
    ]
    vertices.extend(orelha_esq_pontos)
    
    # Ponta PRETA da orelha esquerda
    ponta_preta_esq = [
        [-0.25, -0.05, 3.1], [-0.3, 0.0, 3.15], [-0.25, 0.05, 3.1],
        [-0.2, 0.0, 3.2]
    ]
    vertices.extend(ponta_preta_esq)
    
    # Orelha DIREITA (simétrica)
    orelha_dir_pontos = [
        # Base da orelha
        [0.6, -0.2, 2.2], [0.7, 0.0, 2.2], [0.7, 0.2, 2.2],
        [0.5, -0.15, 2.4], [0.6, 0.0, 2.4], [0.6, 0.15, 2.4],
        # Meio da orelha
        [0.4, -0.1, 2.7], [0.45, 0.0, 2.7], [0.45, 0.1, 2.7],
        # Ponta da orelha
        [0.3, 0.0, 3.0]
    ]
    vertices.extend(orelha_dir_pontos)
    
    # Ponta PRETA da orelha direita
    ponta_preta_dir = [
        [0.25, -0.05, 3.1], [0.3, 0.0, 3.15], [0.25, 0.05, 3.1],
        [0.2, 0.0, 3.2]
    ]
    vertices.extend(ponta_preta_dir)
    
    # =============================================================================
    # 3. CORPO - Pequeno em formato de pêra (como na imagem)
    # =============================================================================
    print("🫃 Criando CORPO (formato pêra)...")
    
    # Corpo menor que a cabeça, formato de pêra
    centro_corpo = [0, 0, -0.2]
    
    for i in range(16):
        for j in range(12):
            u = i / 15.0
            v = j / 11.0
            
            theta = u * 2 * math.pi
            phi = v * math.pi
            
            # Formato de pêra: mais largo embaixo, mais estreito em cima
            raio_x = 0.6 + 0.2 * math.sin(phi)  # Varia com a altura
            raio_y = 0.5 + 0.15 * math.sin(phi)
            raio_z = 0.8
            
            x = centro_corpo[0] + raio_x * math.sin(phi) * math.cos(theta)
            y = centro_corpo[1] + raio_y * math.sin(phi) * math.sin(theta)
            z = centro_corpo[2] + raio_z * math.cos(phi)
            
            vertices.append([x, y, z])
    
    # =============================================================================
    # 4. BRAÇOS - Levantados alegremente (pose da imagem)
    # =============================================================================
    print("💪 Criando BRAÇOS levantados...")
    
    # Braço ESQUERDO (do Pikachu) - levantado
    braco_esq_pontos = [
        # Ombro
        [-0.8, 0.1, 0.5], [-0.9, 0.0, 0.5], [-0.9, -0.1, 0.5],
        # Meio do braço
        [-1.2, 0.0, 0.8], [-1.3, 0.1, 0.8], [-1.3, -0.1, 0.8],
        # Antebraço
        [-1.5, -0.1, 1.2], [-1.6, 0.0, 1.2], [-1.6, 0.1, 1.2],
        # Mão levantada
        [-1.7, 0.0, 1.5], [-1.8, 0.1, 1.6], [-1.8, -0.1, 1.6]
    ]
    vertices.extend(braco_esq_pontos)
    
    # Braço DIREITO - levantado (simétrico)
    braco_dir_pontos = [
        # Ombro
        [0.8, 0.1, 0.5], [0.9, 0.0, 0.5], [0.9, -0.1, 0.5],
        # Meio do braço
        [1.2, 0.0, 0.8], [1.3, 0.1, 0.8], [1.3, -0.1, 0.8],
        # Antebraço
        [1.5, -0.1, 1.2], [1.6, 0.0, 1.2], [1.6, 0.1, 1.2],
        # Mão levantada
        [1.7, 0.0, 1.5], [1.8, 0.1, 1.6], [1.8, -0.1, 1.6]
    ]
    vertices.extend(braco_dir_pontos)
    
    # =============================================================================
    # 5. PERNAS - Curtas e fofas
    # =============================================================================
    print("🦵 Criando PERNAS curtas...")
    
    # Perna ESQUERDA
    perna_esq_pontos = [
        # Coxa
        [-0.3, 0.0, -0.8], [-0.35, 0.1, -0.8], [-0.35, -0.1, -0.8],
        [-0.3, 0.0, -1.2], [-0.35, 0.1, -1.2], [-0.35, -0.1, -1.2],
        # Pé
        [-0.4, 0.15, -1.4], [-0.3, 0.15, -1.4], [-0.35, 0.2, -1.4]
    ]
    vertices.extend(perna_esq_pontos)
    
    # Perna DIREITA
    perna_dir_pontos = [
        # Coxa
        [0.3, 0.0, -0.8], [0.35, 0.1, -0.8], [0.35, -0.1, -0.8],
        [0.3, 0.0, -1.2], [0.35, 0.1, -1.2], [0.35, -0.1, -1.2],
        # Pé
        [0.4, 0.15, -1.4], [0.3, 0.15, -1.4], [0.35, 0.2, -1.4]
    ]
    vertices.extend(perna_dir_pontos)
    
    # =============================================================================
    # 6. RABO - Formato de RAIO característico ⚡
    # =============================================================================
    print("⚡ Criando RABO em formato de raio...")
    
    # Rabo em formato de raio (zigzag característico)
    rabo_pontos = [
        # Base (conecta ao corpo)
        [0.0, -0.7, 0.0],
        # Primeira seção
        [0.2, -1.0, 0.2], [0.3, -1.1, 0.2], [0.1, -1.2, 0.2],
        # Zigzag para a esquerda
        [-0.1, -1.5, 0.4], [0.0, -1.6, 0.4], [-0.2, -1.7, 0.4],
        # Zigzag para a direita
        [0.3, -2.0, 0.6], [0.4, -2.1, 0.6], [0.2, -2.2, 0.6],
        # Seção final
        [0.0, -2.5, 0.8], [0.1, -2.6, 0.8], [-0.1, -2.7, 0.8],
        # Ponta larga do rabo
        [0.3, -2.8, 1.0], [0.4, -2.9, 1.0], [0.2, -3.0, 1.0],
        [0.5, -2.9, 1.1], [0.1, -3.1, 1.1]
    ]
    vertices.extend(rabo_pontos)
    
    # =============================================================================
    # 7. DETALHES FACIAIS - Olhos, bochechas, boca
    # =============================================================================
    print("😊 Adicionando detalhes FACIAIS...")
    
    # OLHOS grandes e expressivos
    vertices.extend([
        # Olho esquerdo
        [-0.25, 0.6, 1.8], [-0.3, 0.65, 1.8], [-0.2, 0.65, 1.8],
        # Olho direito
        [0.25, 0.6, 1.8], [0.3, 0.65, 1.8], [0.2, 0.65, 1.8],
    ])
    
    # BOCHECHAS vermelhas (marcadores)
    vertices.extend([
        [-0.8, 0.4, 1.3], [-0.85, 0.45, 1.3], [-0.75, 0.45, 1.3],  # Esquerda
        [0.8, 0.4, 1.3], [0.85, 0.45, 1.3], [0.75, 0.45, 1.3],     # Direita
    ])
    
    # BOCA sorridente
    vertices.extend([
        [-0.05, 0.7, 1.4], [0.0, 0.75, 1.4], [0.05, 0.7, 1.4],
        [0.0, 0.65, 1.35]  # Língua vermelha
    ])
    
    print(f"✅ Pikachu criado: {len(vertices)} vértices")
    return np.array(vertices)

def gerar_faces_pikachu_otimizado(vertices):
    """
    Gera faces de forma otimizada para melhor visualização
    """
    print("🔗 Gerando faces otimizadas...")
    faces = []
    n = len(vertices)
    
    # Estratégia mais eficiente: conectar pontos em grupos anatômicos
    for i in range(0, n-3, 3):  # Grupos de 3 vértices
        if i+2 < n:
            faces.append([i, i+1, i+2])
    
    # Adicionar algumas conexões entre grupos próximos
    for i in range(n):
        for j in range(i+1, min(i+6, n)):
            for k in range(j+1, min(j+4, n)):
                dist1 = np.linalg.norm(vertices[i] - vertices[j])
                dist2 = np.linalg.norm(vertices[j] - vertices[k])
                dist3 = np.linalg.norm(vertices[k] - vertices[i])
                
                # Apenas triângulos com arestas razoáveis
                if dist1 < 0.5 and dist2 < 0.5 and dist3 < 0.5:
                    faces.append([i, j, k])
    
    print(f"✅ {len(faces)} faces geradas")
    return faces

def visualizar_pikachu_matplotlib(vertices, faces):
    """
    Visualiza o Pikachu usando matplotlib com interatividade
    """
    print("🎨 Iniciando visualização matplotlib...")
    
    # Configurar matplotlib
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('🎮 PIKACHU MESH SUPREMA - Baseado na Imagem Real', 
                fontsize=16, fontweight='bold', color='yellow')
    
    # Plot principal
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotar vértices como pontos
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='gold', s=30, alpha=0.8, label='Vértices Pikachu')
    
    # Plotar faces como wireframe
    print("📐 Renderizando faces...")
    for i, face in enumerate(faces[:min(1000, len(faces))]):  # Limitar para performance
        if i % 100 == 0:
            print(f"   Renderizando face {i}/{min(1000, len(faces))}...")
        
        try:
            p1, p2, p3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Linhas do triângulo
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   'cyan', alpha=0.3, linewidth=0.5)
            ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], 
                   'cyan', alpha=0.3, linewidth=0.5)
            ax.plot([p3[0], p1[0]], [p3[1], p1[1]], [p3[2], p1[2]], 
                   'cyan', alpha=0.3, linewidth=0.5)
        except:
            continue
    
    # Destacar partes anatômicas com cores diferentes
    n_vertices = len(vertices)
    
    # Cabeça (primeiros vértices)
    cabeca_end = 500 if n_vertices > 500 else n_vertices//2
    ax.scatter(vertices[:cabeca_end, 0], vertices[:cabeca_end, 1], vertices[:cabeca_end, 2], 
               c='yellow', s=40, alpha=0.9, label='Cabeça')
    
    # Detalhes faciais (últimos vértices)
    if n_vertices > 20:
        ax.scatter(vertices[-20:, 0], vertices[-20:, 1], vertices[-20:, 2], 
                   c='red', s=60, alpha=1.0, label='Detalhes Faciais')
    
    # Configurar eixos
    ax.set_xlabel('X (Largura)', color='white')
    ax.set_ylabel('Y (Profundidade)', color='white')
    ax.set_zlabel('Z (Altura)', color='white')
    
    # Ajustar limites para melhor visualização
    margin = 0.2
    x_range = vertices[:, 0].max() - vertices[:, 0].min()
    y_range = vertices[:, 1].max() - vertices[:, 1].min()
    z_range = vertices[:, 2].max() - vertices[:, 2].min()
    
    ax.set_xlim(vertices[:, 0].min() - margin * x_range, 
               vertices[:, 0].max() + margin * x_range)
    ax.set_ylim(vertices[:, 1].min() - margin * y_range, 
               vertices[:, 1].max() + margin * y_range)
    ax.set_zlim(vertices[:, 2].min() - margin * z_range, 
               vertices[:, 2].max() + margin * z_range)
    
    # Personalizar aparência
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.legend()
    
    # Informações no plot
    info_text = f"""
📊 ESTATÍSTICAS:
   Vértices: {len(vertices):,}
   Faces: {len(faces):,}
   
🎨 CARACTERÍSTICAS:
   ✅ Cabeça dominante
   ✅ Orelhas pontiagudas
   ✅ Braços levantados
   ✅ Corpo formato pêra
   ✅ Rabo em raio ⚡
   
🎮 CONTROLES:
   Mouse: Rotacionar
   Scroll: Zoom
   Botões: Navegação
"""
    
    plt.figtext(0.02, 0.02, info_text, fontsize=8, color='lightgreen', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    print("🚀 Mostrando visualização matplotlib...")
    plt.show()

def salvar_pikachu_obj(vertices, faces, arquivo):
    """Salva mesh em formato OBJ"""
    print(f"💾 Salvando: {arquivo}")
    
    with open(arquivo, 'w') as f:
        f.write("# PIKACHU MESH SUPREMA - Baseado na imagem exata\n")
        f.write("# Proporções e anatomia da imagem fornecida\n\n")
        
        # Vértices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Faces
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"✅ Salvo: {arquivo}")

def main():
    """Função principal"""
    print("🎮" + "="*60)
    print("    PIKACHU MESH SUPREMA - BASEADO NA IMAGEM EXATA")
    print("    Proporções e anatomia da imagem + Matplotlib")
    print("="*64)
    
    try:
        # Criar mesh baseada na imagem
        vertices = criar_pikachu_imagem_exata()
        
        # Gerar faces
        faces = gerar_faces_pikachu_otimizado(vertices)
        
        # Salvar
        arquivo = "pikachu_mesh_suprema.obj"
        salvar_pikachu_obj(vertices, faces, arquivo)
        
        # Estatísticas
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   🎯 Vértices: {len(vertices):,}")
        print(f"   🔺 Faces: {len(faces):,}")
        
        # Dimensões anatômicas
        x_range = vertices[:, 0].max() - vertices[:, 0].min()
        y_range = vertices[:, 1].max() - vertices[:, 1].min()
        z_range = vertices[:, 2].max() - vertices[:, 2].min()
        
        print(f"\n📏 DIMENSÕES:")
        print(f"   Largura: {x_range:.2f}")
        print(f"   Profundidade: {y_range:.2f}")
        print(f"   Altura: {z_range:.2f}")
        
        print(f"\n🎨 CARACTERÍSTICAS DA IMAGEM:")
        print(f"   ✅ Cabeça dominante (60% da figura)")
        print(f"   ✅ Orelhas pontiagudas inclinadas")
        print(f"   ✅ Braços levantados alegremente")
        print(f"   ✅ Corpo pequeno formato pêra")
        print(f"   ✅ Rabo em formato de raio")
        print(f"   ✅ Detalhes faciais expressivos")
        
        # VISUALIZAR COM MATPLOTLIB
        print(f"\n🚀 INICIANDO VISUALIZAÇÃO MATPLOTLIB...")
        visualizar_pikachu_matplotlib(vertices, faces)
        
        print("="*64)
        print("🏆 PIKACHU MESH SUPREMA CRIADA E VISUALIZADA!")
        
    except ImportError:
        print("❌ Matplotlib não encontrado!")
        print("💡 Instale com: pip install matplotlib")
        print("📁 Mesh salva em: pikachu_mesh_suprema.obj")
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("📁 Verificando se mesh foi salva...")

if __name__ == "__main__":
    main()
