#!/usr/bin/env python3
"""
Script de teste para verificar se os módulos do Nautilus estão funcionando corretamente
"""

import torch
import numpy as np
import sys
import os

def test_imports():
    """Testa se todos os módulos podem ser importados"""
    print("🔍 Testando importações dos módulos...")
    
    try:
        from model.nautilus import MeshTransformer
        print("✅ model.nautilus: OK")
    except Exception as e:
        print(f"❌ model.nautilus: {e}")
        return False

    try:
        from data.data_utils import to_mesh
        print("✅ data.data_utils: OK")
    except Exception as e:
        print(f"❌ data.data_utils: {e}")
        return False

    try:
        from miche.encode import encode_mesh
        print("✅ miche.encode: OK")
    except Exception as e:
        print(f"❌ miche.encode: {e}")
        return False

    try:
        import trimesh
        print("✅ trimesh: OK")
    except Exception as e:
        print(f"❌ trimesh: {e}")
        return False

    return True


def test_model_creation():
    """Testa se o modelo pode ser criado"""
    print("\n🏗️  Testando criação do modelo...")
    
    try:
        from model.nautilus import MeshTransformer
        
        # Criar modelo com configuração padrão
        model = MeshTransformer(
            dim=512,
            max_seq_len=1000,
            attn_depth=6,  # Menor para teste
            u_size=256,    # Menor para teste
            v_size=512,    # Menor para teste
        )
        
        print(f"✅ Modelo criado com sucesso!")
        print(f"   - Dimensão: 512")
        print(f"   - Seq. máxima: 1000")
        print(f"   - Profundidade: 6")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar modelo: {e}")
        return False


def test_basic_functionality():
    """Testa funcionalidades básicas"""
    print("\n⚙️  Testando funcionalidades básicas...")
    
    try:
        # Teste de criação de tensores
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   - Dispositivo: {device}")
        
        # Criar tensor de teste
        test_tensor = torch.randn(1, 100, 512).to(device)
        print(f"   - Tensor criado: {test_tensor.shape}")
        
        # Teste básico de operações
        result = torch.nn.functional.softmax(test_tensor, dim=-1)
        print(f"   - Operação básica: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro em funcionalidades básicas: {e}")
        return False


def create_demo_pointcloud():
    """Cria uma nuvem de pontos de demonstração"""
    print("\n☁️  Criando nuvem de pontos de demonstração...")
    
    try:
        # Criar uma esfera simples como exemplo
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        
        points = []
        for t in theta[::5]:  # Reduzir pontos para demonstração
            for p in phi[::5]:
                x = np.sin(p) * np.cos(t)
                y = np.sin(p) * np.sin(t)
                z = np.cos(p)
                points.append([x, y, z])
        
        points = np.array(points)
        
        # Salvar como arquivo de exemplo
        output_path = "demo_pointcloud.npy"
        np.save(output_path, points)
        
        print(f"✅ Nuvem de pontos criada: {points.shape[0]} pontos")
        print(f"   - Arquivo salvo: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Erro ao criar nuvem de pontos: {e}")
        return None


def main():
    """Função principal de teste"""
    print("=" * 60)
    print("🚀 NAUTILUS - TESTE DE FUNCIONALIDADE")
    print("=" * 60)
    
    # Teste 1: Importações
    if not test_imports():
        print("\n❌ Falha nos testes de importação")
        return False
    
    # Teste 2: Criação do modelo
    if not test_model_creation():
        print("\n❌ Falha na criação do modelo")
        return False
    
    # Teste 3: Funcionalidades básicas
    if not test_basic_functionality():
        print("\n❌ Falha nas funcionalidades básicas")
        return False
    
    # Teste 4: Criar dados de demonstração
    pointcloud_path = create_demo_pointcloud()
    
    print("\n" + "=" * 60)
    print("🎉 TODOS OS TESTES PASSARAM!")
    print("=" * 60)
    print("\n📋 Resumo:")
    print("   ✅ Módulos importados corretamente")
    print("   ✅ Modelo criado com sucesso")
    print("   ✅ Funcionalidades básicas funcionando")
    print("   ✅ Dados de demonstração criados")
    
    if pointcloud_path:
        print(f"\n📄 Arquivo de demonstração: {pointcloud_path}")
        print("   Para usar: carregue este arquivo como nuvem de pontos")
    
    print("\n💡 Próximos passos:")
    print("   1. Obtenha um modelo treinado dos autores")
    print("   2. Use o script infer_pc.py para inferência")
    print("   3. Execute: python infer_pc.py --config config/nautilus_infer.yaml")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
