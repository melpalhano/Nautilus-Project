#!/usr/bin/env python3
"""
Teste simples de correção do Pikachu
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def teste_simples():
    print("🔧 TESTE SIMPLES DE CORREÇÃO...")
    
    try:
        # Tenta carregar a imagem
        print("📷 Carregando pikachu.png...")
        image = Image.open("figures/pikachu.png").convert('RGBA')
        img_array = np.array(image)
        
        print(f"   ✅ Imagem carregada: {image.size}")
        print(f"   📊 Shape do array: {img_array.shape}")
        
        # Cria máscara simples
        if img_array.shape[2] == 4:  # Tem canal alpha
            alpha = img_array[:, :, 3]
            mask = alpha > 128
        else:
            # Remove fundo branco
            img_rgb = img_array[:, :, :3]
            white_mask = np.all(img_rgb > 240, axis=2)
            mask = ~white_mask
        
        print(f"   🎯 Pixels do Pikachu: {np.sum(mask)} / {mask.size}")
        
        # Gera pontos simples
        h, w = mask.shape
        points = []
        
        step = 10
        for y in range(0, h, step):
            for x in range(0, w, step):
                if mask[y, x]:
                    x_norm = (x / w) * 2 - 1
                    y_norm = -(y / h) * 2 + 1
                    z = np.random.uniform(-0.1, 0.1)
                    points.append([x_norm, y_norm, z])
        
        points = np.array(points)
        print(f"   ☁️ Point cloud: {len(points)} pontos")
        
        # Visualização simples
        fig = plt.figure(figsize=(12, 6))
        
        # Imagem original
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(image)
        ax1.set_title('Pikachu Original')
        ax1.axis('off')
        
        # Point cloud
        ax2 = plt.subplot(1, 2, 2, projection='3d')
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='yellow', s=2, alpha=0.8)
        ax2.set_title('Point Cloud Simples')
        
        plt.tight_layout()
        plt.savefig('pikachu_teste_simples.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Visualização salva: pikachu_teste_simples.png")
        print("\n🎉 TESTE SIMPLES CONCLUÍDO!")
        
    except Exception as e:
        print(f"   ❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    teste_simples()
