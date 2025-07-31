#!/usr/bin/env python3
"""
Execução controlada do Pikachu corrigido
"""

# Importações necessárias
import sys
import os

# Adiciona o diretório atual ao path
sys.path.append(os.getcwd())

def executar_com_controle():
    """Executa a correção com controle de erros"""
    try:
        print("🚀 INICIANDO CORREÇÃO CONTROLADA...")
        
        # Importa o módulo
        import corrigir_pikachu
        print("   ✅ Módulo importado")
        
        # Executa a função principal
        print("   🔧 Executando correção...")
        corrigir_pikachu.corrigir_pikachu()
        
        print("   🎉 CONCLUÍDO!")
        
    except Exception as e:
        print(f"   ❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    executar_com_controle()
