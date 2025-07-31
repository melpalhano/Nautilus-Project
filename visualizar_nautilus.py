#!/usr/bin/env python3
"""
Visualização e demonstração completa do projeto Nautilus
Mostra o que é, para que serve e como funciona visualmente
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import yaml
import os

def explicar_projeto_nautilus():
    """Explica o que é o projeto Nautilus"""
    print("🌊 PROJETO NAUTILUS - VISÃO GERAL")
    print("=" * 50)
    
    print("\n🎯 O QUE É:")
    print("   O Nautilus é um sistema de IA que converte nuvens de pontos 3D")
    print("   em malhas 3D (meshes) de alta qualidade usando transformers.")
    
    print("\n🔄 PROCESSO:")
    print("   Nuvem de Pontos → Encoder → Transformer → Decoder → Mesh 3D")
    print("   ☁️ Pontos 3D    → 🧠 IA   → 🔀 Processa → 🎨 Gera → 🗿 Objeto")
    
    print("\n💡 PARA QUE SERVE:")
    print("   • Reconstrução 3D a partir de scans")
    print("   • Geração de modelos 3D para jogos")
    print("   • Modelagem automática para arquitetura")
    print("   • Prototipagem rápida e impressão 3D")
    print("   • Realidade virtual e aumentada")
    
    print("\n🏆 DIFERENCIAIS:")
    print("   • Localidade-aware: Entende estruturas locais")
    print("   • Escalonável: Funciona com meshes grandes")
    print("   • Alta qualidade: Meshes detalhadas")
    print("   • Rápido: ~3-4 minutos para 5000 faces")

def criar_visualizacao_exemplo():
    """Cria visualização do processo Nautilus"""
    print("\n📊 CRIANDO VISUALIZAÇÃO DO PROCESSO...")
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('NAUTILUS: De Nuvem de Pontos para Mesh 3D', fontsize=16, fontweight='bold')
    
    # 1. Nuvem de pontos original (esfera)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Gerar pontos de uma esfera
    n_points = 1000
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    ax1.scatter(x, y, z, c='blue', s=1, alpha=0.6)
    ax1.set_title('1. Nuvem de Pontos\n(Input)', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 2. Processo de encoding
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.text(0.5, 0.7, '2. Encoding', ha='center', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.5, '🧠 Michelangelo Encoder', ha='center', fontsize=12)
    ax2.text(0.5, 0.4, '↓', ha='center', fontsize=20)
    ax2.text(0.5, 0.3, 'Tokens: [u₁, v₁, u₂, v₂, ...]', ha='center', fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # 3. Transformer processing
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.text(0.5, 0.7, '3. Transformer', ha='center', fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.5, '🔀 MeshTransformer', ha='center', fontsize=12)
    ax3.text(0.5, 0.4, '24 layers, 1024 dim', ha='center', fontsize=10)
    ax3.text(0.5, 0.3, 'Self-attention + FFN', ha='center', fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # 4. Mesh resultante (wireframe)
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    
    # Criar mesh simples (icosaedro)
    t = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    
    # Normalizar
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Plotar vértices
    ax4.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='red', s=50)
    
    # Conectar alguns vértices para mostrar estrutura
    connections = [(0, 1), (0, 5), (0, 7), (0, 10), (0, 11)]
    for start, end in connections:
        ax4.plot([vertices[start, 0], vertices[end, 0]],
                [vertices[start, 1], vertices[end, 1]],
                [vertices[start, 2], vertices[end, 2]], 'r-', alpha=0.7)
    
    ax4.set_title('4. Mesh 3D\n(Output)', fontweight='bold')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
    # 5. Estatísticas
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.text(0.5, 0.8, '5. Estatísticas', ha='center', fontsize=14, fontweight='bold')
    ax5.text(0.1, 0.6, f'• Pontos input: {n_points}', fontsize=10)
    ax5.text(0.1, 0.5, f'• Vértices mesh: {len(vertices)}', fontsize=10)
    ax5.text(0.1, 0.4, '• Tempo: ~3-4 min', fontsize=10)
    ax5.text(0.1, 0.3, '• GPU: A100', fontsize=10)
    ax5.text(0.1, 0.2, '• Faces: até 5000', fontsize=10)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    
    # 6. Aplicações
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.5, 0.8, '6. Aplicações', ha='center', fontsize=14, fontweight='bold')
    ax6.text(0.1, 0.6, '🎮 Jogos', fontsize=10)
    ax6.text(0.1, 0.5, '🏗️ Arquitetura', fontsize=10)
    ax6.text(0.1, 0.4, '🖨️ Impressão 3D', fontsize=10)
    ax6.text(0.1, 0.3, '🥽 VR/AR', fontsize=10)
    ax6.text(0.1, 0.2, '🔬 Pesquisa', fontsize=10)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Salvar visualização
    output_path = 'nautilus_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualização salva: {output_path}")
    
    # Mostrar se possível
    try:
        plt.show()
        print("✅ Visualização exibida")
    except:
        print("ℹ️ Para ver a imagem, abra: nautilus_visualization.png")
    
    return output_path

def mostrar_arquitetura_tecnica():
    """Mostra detalhes técnicos da arquitetura"""
    print("\n🏗️ ARQUITETURA TÉCNICA DO NAUTILUS")
    print("=" * 45)
    
    try:
        # Carregar configuração
        with open('config/nautilus_infer.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("📋 CONFIGURAÇÃO DO MODELO:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        print("\n🧠 COMPONENTES PRINCIPAIS:")
        print("   1. Michelangelo Encoder (encoder_name)")
        print("      • Converte nuvem de pontos → features")
        print("      • Dimensão: 256 features")
        print("   ")
        print("   2. MeshTransformer")
        print(f"      • Dimensão: {config['dim']}")
        print(f"      • Profundidade: {config['depth']} layers")
        print(f"      • Seq. máxima: {config['max_seq_len']}")
        print(f"      • Vocabulário U: {config['u_size']}")
        print(f"      • Vocabulário V: {config['v_size']}")
        print("   ")
        print("   3. Coordinate Compression")
        print(f"      • Quantização: {config['quant_bit']} bits")
        print("      • Compressão de coordenadas 3D")
        
        # Calcular modelo
        from model.nautilus import MeshTransformer
        model = MeshTransformer(
            dim=config['dim'],
            max_seq_len=config['max_seq_len'],
            attn_depth=config['depth'],
            u_size=config['u_size'],
            v_size=config['v_size']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n📊 ESTATÍSTICAS DO MODELO:")
        print(f"   • Total de parâmetros: {total_params:,}")
        print(f"   • Tamanho em memória: ~{total_params * 4 / 1e6:.1f} MB")
        print(f"   • Arquitetura: Transformer decoder")
        print(f"   • Atenção: Multi-head self-attention")
        
    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")

def criar_exemplo_dados():
    """Cria e salva exemplos de dados para visualização"""
    print("\n💾 CRIANDO EXEMPLOS DE DADOS...")
    
    # 1. Nuvem de pontos - Esfera
    print("   📊 Criando nuvem de pontos (esfera)...")
    n_points = 2048
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    sphere_points = np.stack([x, y, z], axis=1)
    np.save('exemplo_esfera_points.npy', sphere_points)
    print(f"      ✅ Salvo: exemplo_esfera_points.npy ({sphere_points.shape[0]} pontos)")
    
    # 2. Nuvem de pontos - Cubo
    print("   📦 Criando nuvem de pontos (cubo)...")
    cube_points = []
    
    # Faces do cubo
    for face in range(6):
        for _ in range(300):
            if face == 0:  # face frontal
                cube_points.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 1])
            elif face == 1:  # face traseira
                cube_points.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1), -1])
            elif face == 2:  # face direita
                cube_points.append([1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            elif face == 3:  # face esquerda
                cube_points.append([-1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            elif face == 4:  # face superior
                cube_points.append([np.random.uniform(-1, 1), 1, np.random.uniform(-1, 1)])
            else:  # face inferior
                cube_points.append([np.random.uniform(-1, 1), -1, np.random.uniform(-1, 1)])
    
    cube_points = np.array(cube_points)
    np.save('exemplo_cubo_points.npy', cube_points)
    print(f"      ✅ Salvo: exemplo_cubo_points.npy ({cube_points.shape[0]} pontos)")
    
    # 3. Nuvem de pontos - Cilindro
    print("   🥫 Criando nuvem de pontos (cilindro)...")
    cylinder_points = []
    
    for _ in range(1500):
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.8, 1.0)  # cilindro oco
        z = np.random.uniform(-1, 1)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        cylinder_points.append([x, y, z])
    
    cylinder_points = np.array(cylinder_points)
    np.save('exemplo_cilindro_points.npy', cylinder_points)
    print(f"      ✅ Salvo: exemplo_cilindro_points.npy ({cylinder_points.shape[0]} pontos)")
    
    return ['exemplo_esfera_points.npy', 'exemplo_cubo_points.npy', 'exemplo_cilindro_points.npy']

def mostrar_como_visualizar():
    """Mostra como visualizar os resultados"""
    print("\n👁️ COMO VISUALIZAR O NAUTILUS")
    print("=" * 35)
    
    print("📁 ARQUIVOS CRIADOS:")
    print("   • nautilus_visualization.png - Diagrama do processo")
    print("   • exemplo_*_points.npy - Nuvens de pontos de teste")
    print("   • models/nautilus_dummy_small.pt - Modelo simulado")
    
    print("\n🖥️ FERRAMENTAS DE VISUALIZAÇÃO:")
    print("   1. MATPLOTLIB (Python):")
    print("      • Visualizar nuvens de pontos")
    print("      • Gráficos 3D interativos")
    print("   ")
    print("   2. MESHLAB (Software gratuito):")
    print("      • https://www.meshlab.net/")
    print("      • Visualizar arquivos .ply/.obj")
    print("      • Análise de qualidade de mesh")
    print("   ")
    print("   3. BLENDER (Software gratuito):")
    print("      • https://www.blender.org/")
    print("      • Visualização avançada")
    print("      • Edição de meshes")
    print("   ")
    print("   4. CLOUDCOMPARE (Para nuvens de pontos):")
    print("      • https://www.danielgm.net/cc/")
    print("      • Especializado em point clouds")
    
    print("\n🚀 COMANDOS PARA TESTAR:")
    print("   # Visualizar nuvem de pontos")
    print("   python -c \"")
    print("   import numpy as np")
    print("   import matplotlib.pyplot as plt")
    print("   from mpl_toolkits.mplot3d import Axes3D")
    print("   ")
    print("   points = np.load('exemplo_esfera_points.npy')")
    print("   fig = plt.figure()")
    print("   ax = fig.add_subplot(111, projection='3d')")
    print("   ax.scatter(points[:,0], points[:,1], points[:,2], s=1)")
    print("   plt.show()\"")
    
    print("\n📊 EXECUTAR PIPELINE COMPLETO:")
    print("   python infer_pc_simulado.py \\")
    print("     --config config/nautilus_infer.yaml \\")
    print("     --model_path models/nautilus_dummy_small.pt \\")
    print("     --pc_path exemplo_esfera_points.npy")

def main():
    """Função principal"""
    print("🌊 DEMONSTRAÇÃO VISUAL COMPLETA - NAUTILUS")
    print("=" * 55)
    
    # 1. Explicar o projeto
    explicar_projeto_nautilus()
    
    # 2. Criar visualização
    image_path = criar_visualizacao_exemplo()
    
    # 3. Mostrar arquitetura
    mostrar_arquitetura_tecnica()
    
    # 4. Criar dados de exemplo
    example_files = criar_exemplo_dados()
    
    # 5. Como visualizar
    mostrar_como_visualizar()
    
    print("\n" + "=" * 55)
    print("🎉 DEMONSTRAÇÃO COMPLETA CRIADA!")
    print("=" * 55)
    print("📁 ARQUIVOS PARA VISUALIZAÇÃO:")
    print(f"   🖼️ {image_path}")
    for file in example_files:
        print(f"   ☁️ {file}")
    
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("   1. Abrir nautilus_visualization.png para ver o processo")
    print("   2. Executar comandos de visualização das nuvens de pontos")
    print("   3. Testar pipeline com infer_pc_simulado.py")
    print("   4. Contatar autores para modelo real")
    
    print("\n✨ RESULTADO:")
    print("   Agora você tem uma visão completa e visual do Nautilus!")
    print("=" * 55)

if __name__ == "__main__":
    main()
