#!/usr/bin/env python3
"""
Visualizador Matplotlib Avançado para Meshes do Pikachu
Criado especialmente para visualizar as meshes geradas pelo algoritmo Nautilus
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys

def carregar_mesh_obj(arquivo_obj):
    """
    Carrega uma mesh OBJ e retorna vértices e faces
    """
    vertices = []
    faces = []
    
    try:
        with open(arquivo_obj, 'r') as f:
            for linha in f:
                linha = linha.strip()
                if linha.startswith('v '):
                    # Vértice: v x y z
                    partes = linha.split()
                    if len(partes) >= 4:
                        x, y, z = float(partes[1]), float(partes[2]), float(partes[3])
                        vertices.append([x, y, z])
                elif linha.startswith('f '):
                    # Face: f v1 v2 v3 (ou mais vértices)
                    partes = linha.split()[1:]  # Remove o 'f'
                    face = []
                    for parte in partes:
                        # OBJ usa índices baseados em 1, converter para base 0
                        vertice_idx = int(parte.split('/')[0]) - 1
                        face.append(vertice_idx)
                    if len(face) >= 3:  # Apenas faces válidas
                        faces.append(face)
    except Exception as e:
        print(f"❌ Erro ao carregar {arquivo_obj}: {e}")
        return None, None
    
    if not vertices:
        print(f"❌ Nenhum vértice encontrado em {arquivo_obj}")
        return None, None
    
    return np.array(vertices), faces

def plotar_mesh_3d(vertices, faces, titulo="Mesh do Pikachu", ax=None):
    """
    Plota uma mesh 3D usando matplotlib
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plotar vértices como pontos
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=20, alpha=0.6, label='Vértices')
    
    # Plotar faces como linhas conectando os vértices
    for face in faces:
        if len(face) >= 3:
            # Criar triângulos ou polígonos
            for i in range(len(face)):
                p1 = vertices[face[i]]
                p2 = vertices[face[(i + 1) % len(face)]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                       'b-', alpha=0.3, linewidth=0.5)
    
    # Configurar o plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(titulo)
    ax.legend()
    
    # Ajustar os limites para melhor visualização
    if len(vertices) > 0:
        margin = 0.1
        x_range = vertices[:, 0].max() - vertices[:, 0].min()
        y_range = vertices[:, 1].max() - vertices[:, 1].min()
        z_range = vertices[:, 2].max() - vertices[:, 2].min()
        
        ax.set_xlim(vertices[:, 0].min() - margin * x_range, 
                   vertices[:, 0].max() + margin * x_range)
        ax.set_ylim(vertices[:, 1].min() - margin * y_range, 
                   vertices[:, 1].max() + margin * y_range)
        ax.set_zlim(vertices[:, 2].min() - margin * z_range, 
                   vertices[:, 2].max() + margin * z_range)
    
    return ax

def visualizar_mesh_pikachu(arquivo_mesh):
    """
    Visualiza uma única mesh do Pikachu
    """
    if not os.path.exists(arquivo_mesh):
        print(f"❌ Arquivo não encontrado: {arquivo_mesh}")
        return
    
    print(f"🔄 Carregando mesh: {arquivo_mesh}")
    vertices, faces = carregar_mesh_obj(arquivo_mesh)
    
    if vertices is None:
        return
    
    print(f"✅ Mesh carregada:")
    print(f"   📊 Vértices: {len(vertices)}")
    print(f"   🔺 Faces: {len(faces)}")
    
    # Criar visualização
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f'🎮 VISUALIZAÇÃO 3D: {os.path.basename(arquivo_mesh)}', 
                fontsize=16, fontweight='bold', color='yellow')
    
    # Plot principal
    ax = fig.add_subplot(111, projection='3d')
    plotar_mesh_3d(vertices, faces, f"Mesh: {os.path.basename(arquivo_mesh)}", ax)
    
    # Personalizar cores
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.show()

def comparar_meshes_pikachu():
    """
    Compara múltiplas meshes do Pikachu lado a lado
    """
    meshes_principais = [
        "pikachu_mesh_suprema.obj",
        "pikachu_mesh_perfeita.obj", 
        "pikachu_mesh_nautilus_real.obj",
        "pikachu_mesh_convex_hull_otimizado.obj"
    ]
    
    meshes_existentes = []
    dados_meshes = []
    
    for mesh in meshes_principais:
        if os.path.exists(mesh):
            vertices, faces = carregar_mesh_obj(mesh)
            if vertices is not None:
                meshes_existentes.append(mesh)
                dados_meshes.append((vertices, faces))
                print(f"✅ {mesh}: {len(vertices)} vértices, {len(faces)} faces")
    
    if not meshes_existentes:
        print("❌ Nenhuma mesh encontrada!")
        return
    
    # Criar comparação visual
    plt.style.use('dark_background')
    n_meshes = len(meshes_existentes)
    cols = 2
    rows = (n_meshes + 1) // 2
    
    fig = plt.figure(figsize=(20, 10 * rows))
    fig.suptitle('🎮 COMPARAÇÃO DE MESHES DO PIKACHU', 
                fontsize=20, fontweight='bold', color='yellow')
    
    for i, (mesh_name, (vertices, faces)) in enumerate(zip(meshes_existentes, dados_meshes)):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        plotar_mesh_3d(vertices, faces, f"{os.path.basename(mesh_name)}", ax)
        
        # Personalizar cores
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.show()

def menu_interativo():
    """
    Menu interativo para escolher qual mesh visualizar
    """
    print("🎮" + "="*60)
    print("    VISUALIZADOR MATPLOTLIB - MESHES DO PIKACHU")
    print("="*64)
    
    # Listar todas as meshes disponíveis
    meshes_obj = [f for f in os.listdir('.') if f.endswith('.obj') and 'pikachu' in f.lower()]
    
    if not meshes_obj:
        print("❌ Nenhuma mesh OBJ do Pikachu encontrada!")
        return
    
    print("📋 MESHES DISPONÍVEIS:")
    for i, mesh in enumerate(meshes_obj, 1):
        tamanho = os.path.getsize(mesh)
        print(f"  {i:2d}. {mesh} ({tamanho:,} bytes)")
    
    print(f"\n{len(meshes_obj) + 1:2d}. 🔥 Comparar meshes principais")
    print(f"{len(meshes_obj) + 2:2d}. ❌ Sair")
    
    try:
        escolha = input(f"\n🎯 Escolha uma opção (1-{len(meshes_obj) + 2}): ")
        escolha = int(escolha)
        
        if 1 <= escolha <= len(meshes_obj):
            mesh_escolhida = meshes_obj[escolha - 1]
            print(f"\n🚀 Visualizando: {mesh_escolhida}")
            visualizar_mesh_pikachu(mesh_escolhida)
        elif escolha == len(meshes_obj) + 1:
            print("\n🔥 Comparando meshes principais...")
            comparar_meshes_pikachu()
        elif escolha == len(meshes_obj) + 2:
            print("👋 Saindo...")
            return
        else:
            print("❌ Opção inválida!")
    
    except ValueError:
        print("❌ Por favor, insira um número válido!")
    except KeyboardInterrupt:
        print("\n👋 Visualização cancelada pelo usuário")

def main():
    """
    Função principal
    """
    print("🎮 Iniciando Visualizador Matplotlib para Meshes do Pikachu...")
    
    # Verificar se matplotlib está disponível
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__} detectado")
    except ImportError:
        print("❌ Matplotlib não encontrado!")
        print("💡 Instale com: pip install matplotlib")
        return
    
    # Verificar se estamos no diretório correto
    if not any(f.endswith('.obj') for f in os.listdir('.')):
        print("❌ Nenhum arquivo OBJ encontrado no diretório atual!")
        return
    
    # Se um arquivo específico foi passado como argumento
    if len(sys.argv) > 1:
        arquivo_especifico = sys.argv[1]
        if os.path.exists(arquivo_especifico):
            visualizar_mesh_pikachu(arquivo_especifico)
        else:
            print(f"❌ Arquivo não encontrado: {arquivo_especifico}")
        return
    
    # Visualizar a mesh suprema por padrão se não houver argumentos
    if os.path.exists("pikachu_mesh_suprema.obj"):
        print("🏆 Visualizando a Mesh Suprema (Algoritmo Nautilus)...")
        visualizar_mesh_pikachu("pikachu_mesh_suprema.obj")
    else:
        # Caso contrário, mostrar o menu
        menu_interativo()

if __name__ == "__main__":
    main()
