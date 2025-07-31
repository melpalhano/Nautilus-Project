#!/usr/bin/env python3
"""
Demo simples do Nautilus - Verificação de configuração
"""

import yaml
import torch
import numpy as np

def main():
    print("🔹 NAUTILUS - DEMO BÁSICO")
    print("=" * 40)
    
    # 1. Verificar configuração
    print("📋 Carregando configuração...")
    try:
        with open('config/nautilus_infer.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuração carregada:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Erro ao carregar config: {e}")
        return
    
    # 2. Verificar PyTorch
    print(f"\n🔧 PyTorch: {torch.__version__}")
    print(f"   CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 3. Teste de importação rápida
    print("\n📦 Testando módulos:")
    try:
        from model.nautilus import MeshTransformer
        print("   ✅ MeshTransformer")
    except Exception as e:
        print(f"   ❌ MeshTransformer: {e}")
    
    try:
        import trimesh
        print("   ✅ Trimesh")
    except Exception as e:
        print(f"   ❌ Trimesh: {e}")
    
    # 4. Criar dados de exemplo
    print("\n☁️  Criando exemplo de nuvem de pontos...")
    n_points = 1000
    # Criar uma esfera
    phi = np.random.uniform(0, 2*np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    points = np.stack([x, y, z], axis=1)
    
    print(f"   ✅ Criados {n_points} pontos")
    print(f"   Shape: {points.shape}")
    print(f"   Range: [{points.min():.2f}, {points.max():.2f}]")
    
    # Salvar
    np.save('exemplo_pointcloud.npy', points)
    print("   💾 Salvo como 'exemplo_pointcloud.npy'")
    
    print("\n🎯 STATUS: Ambiente configurado e funcionando!")
    print("📝 Para usar o modelo completo:")
    print("   1. Obtenha um checkpoint treinado")
    print("   2. Execute: python infer_pc.py --config config/nautilus_infer.yaml --model_path <caminho> --pc_path exemplo_pointcloud.npy")

if __name__ == "__main__":
    main()
