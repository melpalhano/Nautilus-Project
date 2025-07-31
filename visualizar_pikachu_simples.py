#!/usr/bin/env python3
"""
Visualizador simples para meshes OBJ do Pikachu
Funciona mesmo sem matplotlib - gera visualização em texto
"""

import os

def ler_vertices_obj(arquivo_obj):
    """Lê os vértices de um arquivo OBJ"""
    vertices = []
    try:
        with open(arquivo_obj, 'r') as f:
            for linha in f:
                if linha.startswith('v '):
                    partes = linha.strip().split()
                    if len(partes) >= 4:
                        x, y, z = float(partes[1]), float(partes[2]), float(partes[3])
                        vertices.append((x, y, z))
    except Exception as e:
        print(f"Erro ao ler {arquivo_obj}: {e}")
    return vertices

def analisar_mesh(nome_arquivo):
    """Analisa uma mesh e mostra informações"""
    print(f"\n{'='*60}")
    print(f"ANÁLISE DA MESH: {nome_arquivo}")
    print(f"{'='*60}")
    
    if not os.path.exists(nome_arquivo):
        print(f"❌ Arquivo não encontrado: {nome_arquivo}")
        return
    
    vertices = ler_vertices_obj(nome_arquivo)
    
    if not vertices:
        print("❌ Nenhum vértice encontrado!")
        return
    
    print(f"✅ Total de vértices: {len(vertices)}")
    
    # Calcular estatísticas
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    
    print(f"\n📊 ESTATÍSTICAS DOS VÉRTICES:")
    print(f"  X: min={min(xs):.3f}, max={max(xs):.3f}, média={sum(xs)/len(xs):.3f}")
    print(f"  Y: min={min(ys):.3f}, max={max(ys):.3f}, média={sum(ys)/len(ys):.3f}")
    print(f"  Z: min={min(zs):.3f}, max={max(zs):.3f}, média={sum(zs)/len(zs):.3f}")
    
    # Mostrar alguns vértices como exemplo
    print(f"\n🔍 PRIMEIROS 10 VÉRTICES:")
    for i, (x, y, z) in enumerate(vertices[:10]):
        print(f"  v{i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
    
    if len(vertices) > 10:
        print(f"  ... e mais {len(vertices)-10} vértices")
    
    # Contar faces
    faces = 0
    try:
        with open(nome_arquivo, 'r') as f:
            for linha in f:
                if linha.startswith('f '):
                    faces += 1
    except:
        pass
    
    print(f"\n🔺 Total de faces: {faces}")
    
    # Calcular tamanho do arquivo
    tamanho = os.path.getsize(nome_arquivo)
    print(f"📁 Tamanho do arquivo: {tamanho:,} bytes ({tamanho/1024:.1f} KB)")

def main():
    """Função principal"""
    print("🎮 VISUALIZADOR DE MESHES DO PIKACHU")
    print("=" * 60)
    
    # Lista das principais meshes para analisar
    meshes_principais = [
        "pikachu_mesh_suprema.obj",
        "pikachu_mesh_perfeita.obj", 
        "pikachu_mesh_nautilus_real.obj",
        "pikachu_mesh_convex_hull_otimizado.obj",
        "pikachu_mesh_delaunay_surface.obj"
    ]
    
    print("📋 MESHES DISPONÍVEIS PARA ANÁLISE:")
    for i, mesh in enumerate(meshes_principais, 1):
        existe = "✅" if os.path.exists(mesh) else "❌"
        print(f"  {i}. {existe} {mesh}")
    
    # Analisar cada mesh
    for mesh in meshes_principais:
        if os.path.exists(mesh):
            analisar_mesh(mesh)
    
    print(f"\n{'='*60}")
    print("🎯 RECOMENDAÇÃO: Use o Windows 3D Viewer ou Paint 3D para visualização completa!")
    print("💡 Para abrir no Windows 3D Viewer: clique duplo no arquivo .obj")
    print("🔧 Para Python com matplotlib: instale Python e as bibliotecas necessárias")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
