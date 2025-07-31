#!/usr/bin/env python3
"""
PIKACHU LOW POLY - MALHA POLIGONAL TRIANGULAR
===========================================
Renderização low poly com facetas geométricas visíveis, construído
manualmente com Python, Numpy e Matplotlib.
- Superfícies planas (sem curvas)
- Arestas pretas destacadas
- Visual cristalino facetado
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class PikachuLowPoly:
    """
    Classe para gerar e renderizar um modelo 3D de um Pikachu
    em estilo Low Poly, definindo cada vértice e face manualmente.
    """
    
    def __init__(self):
        """Inicializa as listas que guardarão a geometria do modelo."""
        self.vertices = []  # Lista de pontos [x, y, z]
        self.faces = []     # Lista de triângulos, conectando os vértices
        self.face_colors = [] # Lista com a cor de cada triângulo

    def create_body_and_head(self):
        """
        Cria o corpo e a cabeça como uma única peça facetada, empilhando
        anéis de polígonos com diferentes raios para dar a forma geral.
        """
        body_vertices = []
        
        # Define as camadas do corpo, de baixo para cima.
        # Cada camada é um anel de pontos.
        layers = [
            {'z': -0.8, 'radius': 0.6, 'segments': 8},  # Base do corpo
            {'z': -0.4, 'radius': 0.8, 'segments': 8},  # Meio do corpo (mais largo)
            {'z': 0.2, 'radius': 0.9, 'segments': 10}, # Transição para cabeça
            {'z': 0.8, 'radius': 1.0, 'segments': 12}, # Base da cabeça
            {'z': 1.4, 'radius': 0.8, 'segments': 12}, # Topo da cabeça
            {'z': 1.8, 'radius': 0.4, 'segments': 8}   # Ponto final no topo
        ]
        
        # Gera os vértices para cada camada
        layer_start_indices = [0]
        for layer in layers:
            z, radius, segments = layer['z'], layer['radius'], layer['segments']
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                body_vertices.append([x, y, z])
            layer_start_indices.append(len(body_vertices))

        base_idx = len(self.vertices)
        self.vertices.extend(body_vertices)
        
        # Conecta as camadas com faces triangulares
        for i in range(len(layers) - 1):
            start_idx1 = layer_start_indices[i]
            end_idx1 = layer_start_indices[i+1]
            segs1 = end_idx1 - start_idx1

            start_idx2 = layer_start_indices[i+1]
            end_idx2 = layer_start_indices[i+2]
            segs2 = end_idx2 - start_idx2

            for j in range(segs1):
                v1 = base_idx + start_idx1 + j
                v2 = base_idx + start_idx1 + (j + 1) % segs1
                
                # Conecta com a camada de cima, ajustando para diferentes números de segmentos
                v3_idx_float = (j / segs1) * segs2
                v3 = base_idx + start_idx2 + int(v3_idx_float)
                v4 = base_idx + start_idx2 + (int(v3_idx_float) + 1) % segs2

                self.faces.append([v1, v2, v4])
                self.faces.append([v1, v4, v3])
                self.face_colors.extend(['yellow', 'yellow'])

    def create_ears(self):
        """Cria as orelhas como prismas pontudos."""
        # Orelha Esquerda
        base_l = np.array([-0.7, 0.3, 1.4])
        tip_l = np.array([-1.0, 0.4, 2.8])
        self._create_single_ear(base_l, tip_l)

        # Orelha Direita (espelhada)
        base_r = np.array([0.7, 0.3, 1.4])
        tip_r = np.array([1.0, 0.4, 2.8])
        self._create_single_ear(base_r, tip_r)

    def _create_single_ear(self, base_pos, tip_pos):
        """Função auxiliar para construir uma única orelha."""
        base_idx = len(self.vertices)
        
        # Base da orelha (4 pontos)
        ear_base = [
            base_pos + [0.1, 0.1, 0], base_pos + [-0.1, 0.1, 0],
            base_pos + [-0.1, -0.1, 0], base_pos + [0.1, -0.1, 0]
        ]
        
        # Ponta da orelha (1 ponto) e meio (para a cor preta)
        mid_point = base_pos + 0.7 * (tip_pos - base_pos)
        
        self.vertices.extend(ear_base)
        self.vertices.append(mid_point)
        self.vertices.append(tip_pos)

        # Faces da parte amarela
        for i in range(4):
            v1 = base_idx + i
            v2 = base_idx + (i + 1) % 4
            v3 = base_idx + 4 # Ponto do meio
            self.faces.append([v1, v2, v3])
            self.face_colors.append('yellow')

        # Faces da parte preta (ponta)
        for i in range(4):
            v3 = base_idx + 4 # Ponto do meio
            v4 = base_idx + 5 # Ponta
            v1 = base_idx + i
            # Conecta o meio com a ponta
            self.faces.append([v3, v4, v1])
            self.face_colors.append('black')

    def create_tail(self):
        """Cria o rabo em formato de raio."""
        base_idx = len(self.vertices)
        thickness = 0.15 # Espessura do rabo

        # Define o caminho 2D do rabo
        path = [
            (0, 0), (0.5, 0), (0.5, -0.8), (1.2, -0.8),
            (1.2, -1.6), (1.9, -1.6), (1.9, -1.3)
        ]
        
        # Posição inicial do rabo no corpo
        start_pos = np.array([0.0, -0.8, -0.5])
        
        # Cria os vértices da frente e de trás
        verts_front = [start_pos + [p[1], p[0], thickness/2] for p in path]
        verts_back = [start_pos + [p[1], p[0], -thickness/2] for p in path]
        self.vertices.extend(verts_front)
        self.vertices.extend(verts_back)

        num_pts = len(path)
        # Cria as faces que conectam a frente e as costas
        for i in range(num_pts - 1):
            # Índices dos 4 pontos do segmento
            v1, v2 = base_idx + i, base_idx + i + 1
            v3, v4 = base_idx + num_pts + i, base_idx + num_pts + i + 1
            
            self.faces.append([v1, v2, v4])
            self.faces.append([v1, v4, v3])
            self.face_colors.extend(['yellow', 'yellow'])
            
            # Cor marrom na base do rabo
            if i == 0:
                 self.face_colors[-2:] = ['#A0522D', '#A0522D']


    def create_facial_features(self):
        """Cria os olhos e bochechas como polígonos planos."""
        # Bochecha Esquerda
        self._create_facial_circle(center=[-0.8, 0.4, 1.0], radius=0.25, color='red', segments=8)
        # Bochecha Direita
        self._create_facial_circle(center=[0.8, 0.4, 1.0], radius=0.25, color='red', segments=8)
        # Olho Esquerdo
        self._create_facial_circle(center=[-0.4, 0.6, 1.3], radius=0.15, color='black', segments=8)
        # Olho Direito
        self._create_facial_circle(center=[0.4, 0.6, 1.3], radius=0.15, color='black', segments=8)


    def _create_facial_circle(self, center, radius, color, segments):
        """Função auxiliar para criar um círculo facetado no rosto."""
        base_idx = len(self.vertices)
        center_vtx = np.array(center)
        self.vertices.append(list(center_vtx))

        circle_verts = []
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            # Cria o círculo no plano XZ, e depois rotaciona para o rosto
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            # Adiciona uma pequena profundidade no eixo Y para ficar "na frente"
            circle_verts.append(list(center_vtx + [x, 0.05, z]))
        
        self.vertices.extend(circle_verts)

        # Cria as faces triangulares do círculo
        for i in range(segments):
            v1 = base_idx # Ponto central
            v2 = base_idx + 1 + i
            v3 = base_idx + 1 + (i + 1) % segments
            self.faces.append([v1, v2, v3])
            self.face_colors.append(color)

    def build(self):
        """Chama todas as funções de construção em ordem."""
        print("Construindo geometria do Pikachu Low Poly...")
        self.create_body_and_head()
        self.create_ears()
        self.create_tail()
        self.create_facial_features()
        print(f"Construção finalizada: {len(self.vertices)} vértices, {len(self.faces)} faces.")

    def render(self, elev=20, azim=45):
        """Renderiza o modelo 3D construído usando Matplotlib."""
        if not self.faces:
            print("Modelo vazio. Execute o método .build() primeiro.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')

        # Converte listas para arrays numpy para performance
        vertices_np = np.array(self.vertices)
        
        # Cria a coleção de polígonos
        poly_collection = Poly3DCollection(
            [vertices_np[face] for face in self.faces],
            facecolors=self.face_colors,
            edgecolors='black', # Arestas pretas para o estilo low poly
            linewidths=1.0,
            alpha=1.0
        )
        ax.add_collection3d(poly_collection)

        # Configura os limites dos eixos para enquadrar o modelo
        ax.auto_scale_xyz(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2])
        
        # Remove os eixos e o grid para um visual limpo
        ax.grid(False)
        ax.axis('off')

        # Define o ângulo da câmera
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        plt.show()

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    
    # 1. Cria a instância do nosso construtor
    pikachu = PikachuLowPoly()
    
    # 2. Constrói a geometria do Pikachu
    pikachu.build()
    
    # 3. Renderiza a vista frontal e de costas
    print("Renderizando vista frontal...")
    pikachu.render(elev=10, azim=-25) # Um pouco de lado para ver a forma
    
    print("Renderizando vista de costas...")
    pikachu.render(elev=10, azim=155) # 180 graus de diferença do primeiro
