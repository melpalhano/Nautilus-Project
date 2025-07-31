#!/usr/bin/env python3
"""
Visualizador Simples para Meshes do Pikachu
Versão lightweight que mostra informações das meshes
"""

import os

def analisar_mesh_obj(arquivo):
    """Analisa uma mesh OBJ e mostra estatísticas"""
    if not os.path.exists(arquivo):
        return None
    
    vertices = []
    faces = []
    
    with open(arquivo, 'r') as f:
        for linha in f:
            if linha.startswith('v '):
                partes = linha.strip().split()
                if len(partes) >= 4:
                    x, y, z = float(partes[1]), float(partes[2]), float(partes[3])
                    vertices.append((x, y, z))
            elif linha.startswith('f '):
                faces.append(linha.strip())
    
    return {
        'arquivo': arquivo,
        'vertices': len(vertices),
        'faces': len(faces),
        'tamanho': os.path.getsize(arquivo),
        'pontos': vertices[:10] if vertices else []  # Primeiros 10 pontos
    }

def main():
    """Análise rápida das meshes"""
    print("🎮 ANÁLISE RÁPIDA DAS MESHES DO PIKACHU")
    print("=" * 50)
    
    # Meshes principais para analisar
    meshes = [
        "pikachu_mesh_suprema.obj",
        "pikachu_mesh_perfeita.obj",
        "pikachu_mesh_nautilus_real.obj",
        "pikachu_mesh_convex_hull_otimizado.obj",
        "pikachu_mesh_delaunay_surface.obj"
    ]
    
    for mesh in meshes:
        info = analisar_mesh_obj(mesh)
        if info:
            print(f"\n📁 {mesh}")
            print(f"   📊 Vértices: {info['vertices']:,}")
            print(f"   🔺 Faces: {info['faces']:,}")
            print(f"   💾 Tamanho: {info['tamanho']:,} bytes ({info['tamanho']/1024:.1f} KB)")
            
            if info['pontos']:
                print(f"   🎯 Primeiros pontos:")
                for i, (x, y, z) in enumerate(info['pontos'][:3]):
                    print(f"      v{i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
        else:
            print(f"\n❌ {mesh} - não encontrado")
    
    print("\n" + "=" * 50)
    print("🚀 COMO VISUALIZAR COM MATPLOTLIB:")
    print("1. Execute: instalar_python_matplotlib.bat")
    print("2. Depois: python visualizador_matplotlib_pikachu.py")
    print("\n🏆 MESH RECOMENDADA: pikachu_mesh_suprema.obj")
    print("   (Algoritmo Nautilus que conecta todos os pontos)")

if __name__ == "__main__":
    main()
