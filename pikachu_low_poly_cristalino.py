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
        """Cria o rabo em formato de raio menor, mais grosso e alinhado à esquerda."""
        base_idx = len(self.vertices)
        thickness = 0.4  # Espessura bem maior do rabo

        # Define o caminho 2D do rabo menor e alinhado à esquerda
        path = [
            (-0.2, 0),        # Base no corpo (esquerda)
            (-0.4, 0.3),      # Primeira curva para cima à esquerda
            (-0.1, 0.6),      # Zigue para centro
            (-0.5, 0.9),      # Zague para esquerda
            (-0.3, 1.2)       # Ponta final menor para cima
        ]
        
        # Posição inicial do rabo no corpo (mais para cima e à esquerda)
        start_pos = np.array([-0.3, -0.7, 0.8])  # Posição à esquerda e elevada
        
        # Cria os vértices da frente e de trás com espessura muito maior
        verts_front = [start_pos + [p[0], 0, p[1] + thickness/2] for p in path]
        verts_back = [start_pos + [p[0], 0, p[1] - thickness/2] for p in path]
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
        """Cria apenas os olhos como polígonos planos."""
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
    
    def create_unified_body_head(self):
        """
        Cria corpo-cabeça UNIFICADO como nas imagens
        Conexão suave sem separação visível - CORPO MENOS ARREDONDADO
        """
        print("   🔶 Corpo-cabeça unificado (menos arredondado)...")
        
        # FORMA UNIFICADA: transição suave cabeça→corpo (MAIS ANGULAR)
        vertices = []
        faces = []
        colors = []
        
        resolution = 25
        
        # Criar forma unificada em camadas - CORPO MENOS ARREDONDADO
        layers = [
            # Camada superior (cabeça - mantém formato)
            {'z': 2.5, 'radius': 0.6, 'segments': 8},   # Topo da cabeça
            {'z': 2.2, 'radius': 0.9, 'segments': 8},   # Cabeça média
            {'z': 1.8, 'radius': 1.0, 'segments': 8},   # Cabeça base
            # Transição suave
            {'z': 1.4, 'radius': 0.95, 'segments': 8},  # Transição
            {'z': 1.0, 'radius': 0.85, 'segments': 8},  # Pescoço
            # Corpo - MENOS ARREDONDADO (mais oval/retangular)
            {'z': 0.6, 'radius': 0.7, 'segments': 6},   # Corpo superior (hexagonal)
            {'z': 0.2, 'radius': 0.8, 'segments': 6},   # Corpo médio (mais largo)
            {'z': -0.2, 'radius': 0.75, 'segments': 6}, # Corpo inferior
            {'z': -0.6, 'radius': 0.6, 'segments': 6}   # Base do corpo (menos redondo)
        ]
        
        # Gerar vértices por camada
        for layer in layers:
            z = layer['z']
            radius = layer['radius']
            segments = layer['segments']
            
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                vertices.append([x, y, z])
                colors.append('yellow')
        
        # Gerar faces conectando camadas
        for layer_idx in range(len(layers) - 1):
            segments = layers[layer_idx]['segments']
            
            for i in range(segments):
                next_i = (i + 1) % segments
                
                # Índices dos vértices
                curr_layer_base = layer_idx * segments
                next_layer_base = (layer_idx + 1) * segments
                
                v1 = curr_layer_base + i
                v2 = curr_layer_base + next_i
                v3 = next_layer_base + i
                v4 = next_layer_base + next_i
                
                # Dois triângulos por face
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])
        
        # Adicionar ao objeto
        base_idx = len(self.vertices)
        self.vertices.extend(vertices)
        
        # Ajustar índices das faces
        for face in faces:
            adjusted_face = [v + base_idx for v in face]
            self.faces.append(adjusted_face)
            self.face_colors.append('yellow')
        
        return len(vertices)
    
    def create_pointed_ears(self):
        """
        Orelhas pontiagudas como prismas triangulares
        Terminam em vértice agudo (ponto de cristal)
        """
        print("   🔶 Orelhas cristalinas pontiagudas...")
        
        ear_data = []
        
        # ORELHA ESQUERDA
        ear_base_left = [-0.7, 0, 2.5]
        ear_tip_left = [-0.5, 0, 3.8]
        ear_width = 0.3
        
        # Base da orelha (triângulo)
        base_left = [
            [ear_base_left[0] - ear_width/2, ear_base_left[1] - ear_width/2, ear_base_left[2]],
            [ear_base_left[0] + ear_width/2, ear_base_left[1] - ear_width/2, ear_base_left[2]],
            [ear_base_left[0], ear_base_left[1] + ear_width/2, ear_base_left[2]]
        ]
        
        # Meio da orelha (triângulo menor)
        mid_left = [
            [ear_tip_left[0] - ear_width/4, ear_tip_left[1] - ear_width/4, ear_tip_left[2] - 0.4],
            [ear_tip_left[0] + ear_width/4, ear_tip_left[1] - ear_width/4, ear_tip_left[2] - 0.4],
            [ear_tip_left[0], ear_tip_left[1] + ear_width/4, ear_tip_left[2] - 0.4]
        ]
        
        # Ponta da orelha (vértice único)
        tip_left = ear_tip_left
        
        base_idx = len(self.vertices)
        self.vertices.extend(base_left)
        self.vertices.extend(mid_left)
        self.vertices.append(tip_left)
        
        # Faces da orelha esquerda
        # Seção inferior
        faces_ear = [
            [base_idx, base_idx + 1, base_idx + 3],      # Base → meio
            [base_idx + 1, base_idx + 4, base_idx + 3],
            [base_idx + 1, base_idx + 2, base_idx + 4],
            [base_idx + 2, base_idx + 5, base_idx + 4],
            [base_idx + 2, base_idx, base_idx + 5],
            [base_idx, base_idx + 3, base_idx + 5],
            # Seção superior (meio → ponta)
            [base_idx + 3, base_idx + 4, base_idx + 6],   # Meio → ponta
            [base_idx + 4, base_idx + 5, base_idx + 6],
            [base_idx + 5, base_idx + 3, base_idx + 6]
        ]
        
        self.faces.extend(faces_ear)
        
        # Cores: amarelo na base, preto na ponta
        colors_ear = ['yellow'] * 6 + ['black'] * 3
        self.face_colors.extend(colors_ear)
        
        # ORELHA DIREITA (espelhada)
        ear_base_right = [0.7, 0, 2.5]
        ear_tip_right = [0.5, 0, 3.8]
        
        base_right = [
            [ear_base_right[0] - ear_width/2, ear_base_right[1] - ear_width/2, ear_base_right[2]],
            [ear_base_right[0] + ear_width/2, ear_base_right[1] - ear_width/2, ear_base_right[2]],
            [ear_base_right[0], ear_base_right[1] + ear_width/2, ear_base_right[2]]
        ]
        
        mid_right = [
            [ear_tip_right[0] - ear_width/4, ear_tip_right[1] - ear_width/4, ear_tip_right[2] - 0.4],
            [ear_tip_right[0] + ear_width/4, ear_tip_right[1] - ear_width/4, ear_tip_right[2] - 0.4],
            [ear_tip_right[0], ear_tip_right[1] + ear_width/4, ear_tip_right[2] - 0.4]
        ]
        
        tip_right = ear_tip_right
        
        base_idx_right = len(self.vertices)
        self.vertices.extend(base_right)
        self.vertices.extend(mid_right)
        self.vertices.append(tip_right)
        
        # Faces da orelha direita
        faces_ear_right = [
            [base_idx_right, base_idx_right + 1, base_idx_right + 3],
            [base_idx_right + 1, base_idx_right + 4, base_idx_right + 3],
            [base_idx_right + 1, base_idx_right + 2, base_idx_right + 4],
            [base_idx_right + 2, base_idx_right + 5, base_idx_right + 4],
            [base_idx_right + 2, base_idx_right, base_idx_right + 5],
            [base_idx_right, base_idx_right + 3, base_idx_right + 5],
            [base_idx_right + 3, base_idx_right + 4, base_idx_right + 6],
            [base_idx_right + 4, base_idx_right + 5, base_idx_right + 6],
            [base_idx_right + 5, base_idx_right + 3, base_idx_right + 6]
        ]
        
        self.faces.extend(faces_ear_right)
        self.face_colors.extend(colors_ear)
        
        return 14  # 7 vértices por orelha
    
    def create_prismatic_limbs(self):
        """
        Braços e pernas como prismas simples
        Braços com MÃOS DO PIKACHU (formato correto), pernas curtas
        """
        print("   🔶 Membros com mãos de Pikachu...")
        
        # BRAÇO ESQUERDO com MÃO DO PIKACHU
        arm_left_base = [-0.9, 0, 1.0]
        arm_left_elbow = [-1.1, 0.2, 1.6]
        arm_left_hand = [-1.3, 0.4, 2.0]
        
        # Parte superior do braço (ombro)
        arm_left_vertices = []
        for i in range(6):
            angle = 2 * np.pi * i / 6
            radius = 0.12
            x = arm_left_base[0] + radius * np.cos(angle)
            y = arm_left_base[1] + radius * np.sin(angle)
            z = arm_left_base[2]
            arm_left_vertices.append([x, y, z])
        
        # Cotovelo (articulação)
        for i in range(6):
            angle = 2 * np.pi * i / 6
            radius = 0.10
            x = arm_left_elbow[0] + radius * np.cos(angle)
            y = arm_left_elbow[1] + radius * np.sin(angle)
            z = arm_left_elbow[2]
            arm_left_vertices.append([x, y, z])
        
        # MÃO DO PIKACHU (formato de gota/oval característico)
        hand_width = 0.15
        hand_length = 0.2
        
        # Dedos do Pikachu (3 pontas características)
        finger_positions = [
            [arm_left_hand[0] - hand_width*0.3, arm_left_hand[1] + hand_length, arm_left_hand[2]],     # Dedo esquerdo
            [arm_left_hand[0], arm_left_hand[1] + hand_length*1.2, arm_left_hand[2] + 0.05],          # Dedo central
            [arm_left_hand[0] + hand_width*0.3, arm_left_hand[1] + hand_length, arm_left_hand[2]]      # Dedo direito
        ]
        
        # Base da mão (formato oval)
        hand_base = []
        for i in range(6):
            angle = 2 * np.pi * i / 6
            radius_x = hand_width * 0.7
            radius_y = hand_width * 0.5
            x = arm_left_hand[0] + radius_x * np.cos(angle)
            y = arm_left_hand[1] + radius_y * np.sin(angle)
            z = arm_left_hand[2]
            hand_base.append([x, y, z])
        
        base_idx = len(self.vertices)
        self.vertices.extend(arm_left_vertices)    # Braço (12 vértices)
        self.vertices.extend(hand_base)            # Base da mão (6 vértices)
        self.vertices.extend(finger_positions)     # Dedos (3 vértices)
        
        # Faces do braço esquerdo
        for i in range(6):
            next_i = (i + 1) % 6
            
            # Faces do braço (ombro → cotovelo)
            face1 = [base_idx + i, base_idx + next_i, base_idx + 6 + i]
            face2 = [base_idx + next_i, base_idx + 6 + next_i, base_idx + 6 + i]
            
            self.faces.extend([face1, face2])
            self.face_colors.extend(['yellow', 'yellow'])
        
        # Faces da mão (cotovelo → base da mão)
        for i in range(6):
            next_i = (i + 1) % 6
            
            face1 = [base_idx + 6 + i, base_idx + 6 + next_i, base_idx + 12 + i]
            face2 = [base_idx + 6 + next_i, base_idx + 12 + next_i, base_idx + 12 + i]
            
            self.faces.extend([face1, face2])
            self.face_colors.extend(['yellow', 'yellow'])
        
        # Faces dos dedos (base da mão → dedos)
        for i in range(3):
            finger_idx = base_idx + 18 + i  # Índice do dedo
            hand_vertex1 = base_idx + 12 + (i * 2) % 6     # Vértice da base da mão
            hand_vertex2 = base_idx + 12 + ((i * 2) + 1) % 6
            
            face = [finger_idx, hand_vertex1, hand_vertex2]
            self.faces.append(face)
            self.face_colors.append('yellow')
        
        # BRAÇO DIREITO (espelhado com mesma estrutura de mão)
        arm_right_base = [0.9, 0, 1.0]
        arm_right_elbow = [1.1, 0.2, 1.6]
        arm_right_hand = [1.3, 0.4, 2.0]
        
        arm_right_vertices = []
        
        # Ombro direito
        for i in range(6):
            angle = 2 * np.pi * i / 6
            radius = 0.12
            x = arm_right_base[0] + radius * np.cos(angle)
            y = arm_right_base[1] + radius * np.sin(angle)
            z = arm_right_base[2]
            arm_right_vertices.append([x, y, z])
        
        # Cotovelo direito
        for i in range(6):
            angle = 2 * np.pi * i / 6
            radius = 0.10
            x = arm_right_elbow[0] + radius * np.cos(angle)
            y = arm_right_elbow[1] + radius * np.sin(angle)
            z = arm_right_elbow[2]
            arm_right_vertices.append([x, y, z])
        
        # Mão direita
        finger_positions_right = [
            [arm_right_hand[0] - hand_width*0.3, arm_right_hand[1] + hand_length, arm_right_hand[2]],
            [arm_right_hand[0], arm_right_hand[1] + hand_length*1.2, arm_right_hand[2] + 0.05],
            [arm_right_hand[0] + hand_width*0.3, arm_right_hand[1] + hand_length, arm_right_hand[2]]
        ]
        
        hand_base_right = []
        for i in range(6):
            angle = 2 * np.pi * i / 6
            radius_x = hand_width * 0.7
            radius_y = hand_width * 0.5
            x = arm_right_hand[0] + radius_x * np.cos(angle)
            y = arm_right_hand[1] + radius_y * np.sin(angle)
            z = arm_right_hand[2]
            hand_base_right.append([x, y, z])
        
        base_idx_right = len(self.vertices)
        self.vertices.extend(arm_right_vertices)
        self.vertices.extend(hand_base_right)
        self.vertices.extend(finger_positions_right)
        
        # Faces do braço direito (mesma estrutura)
        for i in range(6):
            next_i = (i + 1) % 6
            
            # Braço
            face1 = [base_idx_right + i, base_idx_right + next_i, base_idx_right + 6 + i]
            face2 = [base_idx_right + next_i, base_idx_right + 6 + next_i, base_idx_right + 6 + i]
            
            # Mão
            face3 = [base_idx_right + 6 + i, base_idx_right + 6 + next_i, base_idx_right + 12 + i]
            face4 = [base_idx_right + 6 + next_i, base_idx_right + 12 + next_i, base_idx_right + 12 + i]
            
            self.faces.extend([face1, face2, face3, face4])
            self.face_colors.extend(['yellow', 'yellow', 'yellow', 'yellow'])
        
        # Dedos direitos
        for i in range(3):
            finger_idx = base_idx_right + 18 + i
            hand_vertex1 = base_idx_right + 12 + (i * 2) % 6
            hand_vertex2 = base_idx_right + 12 + ((i * 2) + 1) % 6
            
            face = [finger_idx, hand_vertex1, hand_vertex2]
            self.faces.append(face)
            self.face_colors.append('yellow')
        
        # PERNAS (prismas curtos)
        leg_positions = [[-0.3, 0, -0.2], [0.3, 0, -0.2]]
        
        for leg_pos in leg_positions:
            leg_vertices = []
            
            # Base da perna
            for i in range(4):  # Prisma quadrado
                angle = 2 * np.pi * i / 4
                radius = 0.15
                x = leg_pos[0] + radius * np.cos(angle)
                y = leg_pos[1] + radius * np.sin(angle)
                z = leg_pos[2]
                leg_vertices.append([x, y, z])
            
            # Ponta da perna
            for i in range(4):
                angle = 2 * np.pi * i / 4
                radius = 0.12
                x = leg_pos[0] + radius * np.cos(angle)
                y = leg_pos[1] + radius * np.sin(angle)
                z = leg_pos[2] - 0.6
                leg_vertices.append([x, y, z])
            
            base_idx_leg = len(self.vertices)
            self.vertices.extend(leg_vertices)
            
            # Faces da perna
            for i in range(4):
                next_i = (i + 1) % 4
                
                face1 = [base_idx_leg + i, base_idx_leg + next_i, base_idx_leg + 4 + i]
                face2 = [base_idx_leg + next_i, base_idx_leg + 4 + next_i, base_idx_leg + 4 + i]
                
                self.faces.extend([face1, face2])
                self.face_colors.extend(['yellow', 'yellow'])
        
        return 12 + 12 + 8 + 8  # Total de vértices dos membros
    
    def create_lightning_tail(self):
        """
        Rabo em formato de raio MAIS ESPESSO - À ESQUERDA
        Posicionado atrás do corpo, com espessura maior
        """
        print("   🔶 Rabo raio MAIS ESPESSO à esquerda...")
        
        # PONTOS DE CONTROLE DO RAIO (POSICIONADO À ESQUERDA)
        lightning_path = [
            [-0.8, -0.9, 0.2],     # Base (lado esquerdo do corpo)
            [-1.2, -1.3, 0.6],     # Primeira curva à esquerda
            [-1.8, -1.6, 1.0],     # Segunda curva
            [-2.2, -1.4, 1.5],     # Terceira curva (subindo)
            [-2.6, -1.0, 2.0],     # Quarta curva
            [-3.0, -0.5, 2.5],     # Quinta curva (ponto alto)
            [-2.8, -0.1, 3.0],     # Sexta curva
            [-3.2, 0.2, 3.3]       # Ponta final
        ]
        
        # CRIAR ESTRUTURA PLANAR DO RAIO MAIS ESPESSO
        tail_vertices = []
        tail_faces = []
        
        # Para cada segmento do raio (ESPESSURA MAIOR)
        for i in range(len(lightning_path) - 1):
            start = lightning_path[i]
            end = lightning_path[i + 1]
            
            # Largura do segmento (consistente e mais espessa)
            width = 0.25  # LARGURA FIXA MAIOR
            thickness = 0.20  # ESPESSURA MUITO MAIOR
            
            # Vértices do segmento (formato de raio mais espesso)
            segment_vertices = [
                # Face frontal (mais espessa)
                [start[0] - width, start[1] - width*0.3, start[2] + thickness],
                [start[0] + width, start[1] + width*0.2, start[2] + thickness],
                [end[0] + width, end[1] + width*0.4, end[2] + thickness],
                [end[0] - width, end[1] - width*0.2, end[2] + thickness],
                # Face traseira (mais espessa)
                [start[0] - width, start[1] - width*0.3, start[2] - thickness],
                [start[0] + width, start[1] + width*0.2, start[2] - thickness],
                [end[0] + width, end[1] + width*0.4, end[2] - thickness],
                [end[0] - width, end[1] - width*0.2, end[2] - thickness]
            ]
            
            base_idx = len(self.vertices)
            self.vertices.extend(segment_vertices)
            tail_vertices.extend(segment_vertices)
            
            # Faces do segmento (mais espesso)
            segment_faces = [
                # Face frontal
                [base_idx, base_idx + 1, base_idx + 2],
                [base_idx, base_idx + 2, base_idx + 3],
                # Face traseira
                [base_idx + 4, base_idx + 6, base_idx + 5],
                [base_idx + 4, base_idx + 7, base_idx + 6],
                # Laterais superiores
                [base_idx, base_idx + 4, base_idx + 5],
                [base_idx, base_idx + 5, base_idx + 1],
                [base_idx + 1, base_idx + 5, base_idx + 6],
                [base_idx + 1, base_idx + 6, base_idx + 2],
                # Laterais inferiores
                [base_idx + 2, base_idx + 6, base_idx + 7],
                [base_idx + 2, base_idx + 7, base_idx + 3],
                [base_idx + 3, base_idx + 7, base_idx + 4],
                [base_idx + 3, base_idx + 4, base_idx]
            ]
            
            self.faces.extend(segment_faces)
            tail_faces.extend(segment_faces)
            
            # Todas as faces do rabo são amarelas
            self.face_colors.extend(['yellow'] * len(segment_faces))
        
        return len(tail_vertices)
    
    def create_facial_features(self):
        """
        Características faciais MAPEADAS E VISÍVEIS: olhos grandes, bochechas destacadas, boca amarela
        """
        print("   🔶 Características faciais MAPEADAS...")
        
        # OLHOS GRANDES E DESTACADOS (mais visíveis)
        eye_positions = [[-0.35, 0.6, 2.4], [0.35, 0.6, 2.4]]  # Mais para frente e maiores
        
        for eye_pos in eye_positions:
            eye_vertices = []
            eye_radius = 0.18  # OLHOS MAIORES
            
            # Criar círculo curvo para o olho (bem visível)
            for i in range(16):  # Mais pontos para melhor definição
                angle = 2 * np.pi * i / 16
                x = eye_pos[0] + eye_radius * np.cos(angle)
                y = eye_pos[1] + eye_radius * np.sin(angle)
                z = eye_pos[2] + 0.05  # Ligeiramente elevado
                eye_vertices.append([x, y, z])
            
            # Centro do olho (mais destacado)
            eye_center = [eye_pos[0], eye_pos[1], eye_pos[2] + 0.08]
            eye_vertices.append(eye_center)
            
            base_idx = len(self.vertices)
            self.vertices.extend(eye_vertices)
            
            # Faces do olho (triângulos do centro) - BEM VISÍVEIS
            for i in range(16):
                next_i = (i + 1) % 16
                face = [base_idx + 16, base_idx + i, base_idx + next_i]  # Centro → borda
                self.faces.append(face)
                self.face_colors.append('black')
        
        # BOCHECHAS VERMELHAS CIRCULARES (MAPEADAS E DESTACADAS)
        cheek_positions = [[-1.1, 0.4, 2.0], [1.1, 0.4, 2.0]]  # Mais nas laterais
        
        for cheek_pos in cheek_positions:
            cheek_vertices = []
            cheek_radius = 0.20  # BOCHECHAS MAIORES E MAIS VISÍVEIS
            
            # Círculo para bochecha (bem definido)
            for i in range(12):  # Mais pontos para melhor definição
                angle = 2 * np.pi * i / 12
                x = cheek_pos[0] + cheek_radius * np.cos(angle)
                y = cheek_pos[1] + cheek_radius * np.sin(angle)
                z = cheek_pos[2] + 0.05  # Ligeiramente elevado
                cheek_vertices.append([x, y, z])
            
            # Centro da bochecha (destacado)
            cheek_center = [cheek_pos[0], cheek_pos[1], cheek_pos[2] + 0.08]
            cheek_vertices.append(cheek_center)
            
            base_idx = len(self.vertices)
            self.vertices.extend(cheek_vertices)
            
            # Faces da bochecha (BEM VISÍVEIS)
            for i in range(12):
                next_i = (i + 1) % 12
                face = [base_idx + 12, base_idx + i, base_idx + next_i]
                self.faces.append(face)
                self.face_colors.append('red')
        
        # BOCA PEQUENA E SORRIDENTE (AMARELA E VISÍVEL)
        mouth_center = [0.0, 0.7, 2.0]  # Mais para frente
        mouth_vertices = []
        
        # Criar pequena curva sorridente (mais definida)
        for i in range(8):  # Mais pontos para melhor curva
            angle = math.pi * (i / 7 - 0.5) * 0.4  # Curva suave
            
            x = mouth_center[0] + 0.12 * math.cos(angle)
            y = mouth_center[1]
            z = mouth_center[2] + 0.05 * math.sin(angle) + 0.05  # Mais elevada
            
            mouth_vertices.append([x, y, z])
        
        # Centro da boca (destacado)
        mouth_center_vertex = [mouth_center[0], mouth_center[1], mouth_center[2] + 0.08]
        mouth_vertices.append(mouth_center_vertex)
        
        base_idx = len(self.vertices)
        self.vertices.extend(mouth_vertices)
        
        # Faces da boca (todas amarelas e visíveis)
        for i in range(7):
            face = [base_idx + 8, base_idx + i, base_idx + i + 1]
            self.faces.append(face)
            self.face_colors.append('yellow')  # BOCA AMARELA
        
        # NARIZ PEQUENO (característica do Pikachu)
        nose_center = [0.0, 0.5, 2.3]
        nose_vertices = [
            [nose_center[0] - 0.03, nose_center[1], nose_center[2]],
            [nose_center[0] + 0.03, nose_center[1], nose_center[2]],
            [nose_center[0], nose_center[1] + 0.05, nose_center[2] + 0.02]
        ]
        
        base_idx_nose = len(self.vertices)
        self.vertices.extend(nose_vertices)
        
        # Face do nariz
        nose_face = [base_idx_nose, base_idx_nose + 1, base_idx_nose + 2]
        self.faces.append(nose_face)
        self.face_colors.append('black')
        
        return 34 + 26 + 9 + 3  # Olhos + bochechas + boca + nariz
    
    def build_complete_low_poly_pikachu(self):
        """
        Constrói Pikachu completo em estilo Low Poly
        """
        print("🔶 CONSTRUINDO PIKACHU LOW POLY...")
        
        # Construir cada parte com coordenadas manuais
        vertices_body = self.create_unified_body_head()
        vertices_ears = self.create_pointed_ears()
        vertices_limbs = self.create_prismatic_limbs()
        vertices_tail = self.create_lightning_tail()
        vertices_face = self.create_facial_features()
        
        total_vertices = len(self.vertices)
        total_faces = len(self.faces)
        
        print(f"✅ LOW POLY Pikachu: {total_vertices} vértices, {total_faces} faces triangulares")
        print(f"   🔶 Corpo-cabeça: unificado suavemente")
        print(f"   🔶 Orelhas: pontiagudas com pontas arredondadas")
        print(f"   🔶 Rabo: posicionado à esquerda")
        print(f"   👀 Olhos: grandes com curva suave")
        print(f"   😊 Boca: pequena e amarela")
    
    def render_single_pikachu(self):
        """
        Renderiza UM ÚNICO Pikachu Low Poly MAIOR em gráfico 3D
        Com anatomia exata e características faciais visíveis
        """
        print("\n🎨 RENDERIZANDO PIKACHU LOW POLY GRANDE EM 3D...")
        
        # Configuração para um único plot MAIOR
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 12), dpi=120)  # FIGURA MAIOR
        fig.patch.set_facecolor('white')
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        
        # Converter vértices para numpy
        vertices = np.array(self.vertices)
        
        # Preparar faces e cores
        triangles = []
        colors = []
        
        color_map = {
            'yellow': [1.0, 0.9, 0.0],    # Amarelo Pikachu
            'black': [0.1, 0.1, 0.1],     # Preto dos olhos
            'red': [0.9, 0.2, 0.2]        # Vermelho das bochechas
        }
        
        for i, face in enumerate(self.faces):
            if len(face) >= 3:
                try:
                    triangle = vertices[face[:3]]
                    triangles.append(triangle)
                    
                    face_color = color_map.get(self.face_colors[i], [1.0, 0.9, 0.0])
                    colors.append(face_color)
                    
                except (IndexError, ValueError):
                    continue
        
        # RENDERIZAR MALHA LOW POLY COM GRID 3D
        if triangles:
            poly_collection = Poly3DCollection(triangles,
                                             facecolors=colors,
                                             edgecolors='black',    # Arestas pretas visíveis
                                             linewidths=1.5,        # Bordas mais destacadas
                                             alpha=0.9,             # Ligeiramente transparente
                                             antialiased=False)     # Sem suavização
            ax.add_collection3d(poly_collection)
        
        # Configurar visualização MAIOR e com GRID 3D
        if len(vertices) > 0:
            margin = 1.0  # MARGEM MAIOR
            
            # Definir limites maiores
            x_min, x_max = vertices[:, 0].min() - margin, vertices[:, 0].max() + margin
            y_min, y_max = vertices[:, 1].min() - margin, vertices[:, 1].max() + margin
            z_min, z_max = vertices[:, 2].min() - margin, vertices[:, 2].max() + margin
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            # ADICIONAR GRID 3D PARA MOSTRAR QUE É 3D
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z', fontsize=12, fontweight='bold')
            
            # Ticks visíveis para mostrar escala 3D
            ax.set_xticks(np.linspace(x_min, x_max, 5))
            ax.set_yticks(np.linspace(y_min, y_max, 5))
            ax.set_zticks(np.linspace(z_min, z_max, 5))
        
        # Ângulo ótimo para mostrar todas as características
        ax.view_init(elev=20, azim=45)
        
        # Título descritivo MAIOR
        fig.suptitle('🔶 PIKACHU LOW POLY 3D\nModelo Triangular • Características Mapeadas • Rabo Espesso', 
                    fontsize=20, fontweight='bold', color='#2c3e50', y=0.95)
        
        # Informações anatômicas detalhadas e DESTACADAS
        info_text = f"""
🎯 PIKACHU LOW POLY 3D - ANATOMIA MAPEADA:

📏 ESTRUTURA CORPORAL:
• Cabeça grande unificada ao corpo (sem pescoço visível)
• Corpo hexagonal menos arredondado (formato angular)
• Braços com mãos de 3 dedos características do Pikachu
• Pernas curtas e robustas (patas típicas)

👂 ORELHAS LONGAS E PONTIAGUDAS:
• Base amarela com prismas triangulares
• Pontas pretas arredondadas elevadas

⚡ RABO RAIO ESPESSO:
• Formato de raio zigzag característico
• Espessura aumentada (não tamanho)
• Posicionado à esquerda do corpo

😊 CARACTERÍSTICAS FACIAIS MAPEADAS:
👀 Olhos: Grandes, pretos e circulares (bem visíveis)
🔴 Bochechas: Círculos vermelhos destacados (bolsas elétricas)
😊 Boca: Pequena, amarela e sorridente
👃 Nariz: Pequeno triângulo preto

📊 ESPECIFICAÇÕES TÉCNICAS 3D:
• Vértices: {len(self.vertices):,}
• Faces triangulares: {len(self.faces):,}
• Estilo: Low Poly Mesh cristalino
• Renderização: 3D com grid visível
• Escala: Modelo ampliado para melhor visualização
"""
        
        fig.text(0.02, 0.02, info_text, fontsize=10, color='#2c3e50',
                bbox=dict(boxstyle="round,pad=0.7", facecolor='#f8f9fa', alpha=0.95, edgecolor='#6c757d'),
                family='monospace', verticalalignment='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.55, left=0.05, right=0.95)
        
        print("✅ Pikachu Low Poly 3D GRANDE renderizado!")
        print(f"   🔶 {len(triangles)} triângulos facetados")
        print(f"   ⚡ Rabo com espessura aumentada")
        print(f"   👀 Características faciais mapeadas e visíveis")
        print(f"   📊 Grid 3D para visualização espacial")
        print(f"   � Modelo ampliado para melhor detalhamento")
        
        plt.show()
    
    def render_three_views(self):
        """
        Renderiza três vistas: frente, lado e costa
        Com barra de gráfico estilo dashboard
        """
        print("\n🎨 RENDERIZANDO TRÊS VISTAS LOW POLY...")
        
        # Configuração para três subplots
        fig = plt.figure(figsize=(18, 6), dpi=100)
        fig.patch.set_facecolor('white')
        
        # Converter vértices para numpy
        vertices = np.array(self.vertices)
        
        # Preparar faces e cores
        triangles = []
        colors = []
        
        color_map = {
            'yellow': [1.0, 0.9, 0.0],
            'black': [0.1, 0.1, 0.1],
            'red': [0.9, 0.2, 0.2]
        }
        
        for i, face in enumerate(self.faces):
            if len(face) >= 3:
                try:
                    triangle = vertices[face[:3]]
                    triangles.append(triangle)
                    
                    face_color = color_map.get(self.face_colors[i], [1.0, 0.9, 0.0])
                    colors.append(face_color)
                    
                except (IndexError, ValueError):
                    continue
        
        # Vista 1: FRENTE (elev=0, azim=0)
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_title('🔶 VISTA FRONTAL', fontsize=14, fontweight='bold', pad=15)
        
        if triangles:
            poly1 = Poly3DCollection(triangles,
                                   facecolors=colors,
                                   edgecolors='black',
                                   linewidths=1.2,
                                   alpha=1.0,
                                   antialiased=False)
            ax1.add_collection3d(poly1)
        
        ax1.view_init(elev=0, azim=0)  # Vista frontal
        self._configure_axis(ax1, vertices)
        
        # Vista 2: LADO (elev=0, azim=90)
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_title('🔶 VISTA LATERAL', fontsize=14, fontweight='bold', pad=15)
        
        if triangles:
            poly2 = Poly3DCollection(triangles,
                                   facecolors=colors,
                                   edgecolors='black',
                                   linewidths=1.2,
                                   alpha=1.0,
                                   antialiased=False)
            ax2.add_collection3d(poly2)
        
        ax2.view_init(elev=0, azim=90)  # Vista lateral
        self._configure_axis(ax2, vertices)
        
        # Vista 3: COSTA (elev=0, azim=180)
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.set_title('🔶 VISTA TRASEIRA', fontsize=14, fontweight='bold', pad=15)
        
        if triangles:
            poly3 = Poly3DCollection(triangles,
                                   facecolors=colors,
                                   edgecolors='black',
                                   linewidths=1.2,
                                   alpha=1.0,
                                   antialiased=False)
            ax3.add_collection3d(poly3)
        
        ax3.view_init(elev=0, azim=180)  # Vista traseira
        self._configure_axis(ax3, vertices)
        
        # Título principal com barra de informações
        fig.suptitle('🔶 PIKACHU LOW POLY - TRÊS VISTAS TÉCNICAS\nMalha Poligonal Triangular • Visual Cristalino • Rabo à Direita', 
                    fontsize=16, fontweight='bold', color='#2c3e50', y=0.95)
        
        # Barra de estatísticas (como dashboard)
        stats_text = f"""
📊 ESTATÍSTICAS DO MODELO:
Vértices: {len(self.vertices):,} | Faces: {len(self.faces):,} | Arestas: {len(self.faces)*3:,}
🔶 Corpo: Unificado (cabeça+corpo) | 👂 Orelhas: Pontiagudas | ⚡ Rabo: Posicionado à direita
🎨 Estilo: Low Poly Cristalino | 📐 Malha: Triangular | ⚫ Bordas: Destacadas
"""
        
        fig.text(0.5, 0.02, stats_text, fontsize=10, color='#2c3e50',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8f9fa', alpha=0.95, edgecolor='#6c757d'),
                ha='center', family='monospace')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.2, left=0.05, right=0.95)
        
        print("✅ TRÊS VISTAS renderizadas!")
        print("   🔶 Vista frontal: azim=0°")
        print("   🔶 Vista lateral: azim=90° (rabo visível)")
        print("   🔶 Vista traseira: azim=180°")
        
        plt.show()
    
    def _configure_axis(self, ax, vertices):
        """Configura eixo para visualização Low Poly"""
        if len(vertices) > 0:
            margin = 0.3
            ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
            ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
            ax.set_zlim(vertices[:, 2].min() - margin, vertices[:, 2].max() + margin)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.axis('off')
        ax.set_facecolor('white')
    
    def render_low_poly_style(self):
        """
        Renderiza em estilo Low Poly perfeito
        - Faces amarelas sólidas
        - Arestas pretas destacadas
        - Fundo branco
        - Visual cristalino facetado
        """
        print("\n🎨 RENDERIZANDO LOW POLY STYLE...")
        
        # Configuração para estilo Low Poly
        plt.style.use('default')
        fig = plt.figure(figsize=(14, 10), dpi=100)
        fig.patch.set_facecolor('white')  # Fundo branco
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        
        # Converter vértices para numpy
        vertices = np.array(self.vertices)
        
        # Preparar faces e cores
        triangles = []
        colors = []
        
        # Mapear cores de string para RGB
        color_map = {
            'yellow': [1.0, 0.9, 0.0],    # Amarelo Pikachu
            'black': [0.1, 0.1, 0.1],     # Preto dos olhos/orelhas
            'red': [0.9, 0.2, 0.2]        # Vermelho das bochechas
        }
        
        for i, face in enumerate(self.faces):
            if len(face) >= 3:
                try:
                    triangle = vertices[face[:3]]
                    triangles.append(triangle)
                    
                    # Cor da face
                    face_color = color_map.get(self.face_colors[i], [1.0, 0.9, 0.0])
                    colors.append(face_color)
                    
                except (IndexError, ValueError):
                    continue
        
        # RENDERIZAR FACES SÓLIDAS (amarelas com arestas pretas)
        if triangles:
            poly_collection = Poly3DCollection(triangles,
                                             facecolors=colors,
                                             edgecolors='black',    # ARESTAS PRETAS
                                             linewidths=1.5,        # Arestas bem visíveis
                                             alpha=1.0,             # Totalmente opaco
                                             antialiased=False)     # Sem suavização (Low Poly)
            ax.add_collection3d(poly_collection)
        
        # Configurar visualização
        if len(vertices) > 0:
            margin = 0.5
            
            ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
            ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
            ax.set_zlim(vertices[:, 2].min() - margin, vertices[:, 2].max() + margin)
        
        # Ângulo ótimo para mostrar facetas
        ax.view_init(elev=20, azim=45)
        
        # Remover eixos para foco nas facetas
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.axis('off')
        
        # Título Low Poly
        fig.suptitle('🔶 PIKACHU LOW POLY\nMalha Poligonal Triangular • Visual Cristalino Facetado', 
                    fontsize=18, fontweight='bold', color='#2c3e50', y=0.93)
        
        # Informações do estilo Low Poly
        info_text = f"""
🔶 ESTILO LOW POLY:
✅ Malha poligonal triangular visível
✅ Superfícies planas (sem curvas)
✅ Arestas pretas destacadas
✅ Visual cristalino facetado
✅ Coordenadas manuais precisas

🎯 ANATOMIA FACETADA:
✅ Cabeça geodésica (prisma octogonal)
✅ Corpo formato copo (pirâmide truncada)
✅ Orelhas pontiagudas (prismas triangulares)
✅ Rabo raio 2.5D (polígonos planos)
✅ Olhos hexagonais pretos
✅ Bochechas octogonais vermelhas

📊 ESPECIFICAÇÕES:
• Vértices: {len(self.vertices):,}
• Faces triangulares: {len(self.faces):,}
• Cores: Amarelo + Preto + Vermelho
• Arestas: Pretas destacadas
• Fundo: Branco puro
• Estilo: Cristal lapidado
"""
        
        fig.text(0.02, 0.02, info_text, fontsize=9, color='#2c3e50',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8f9fa', alpha=0.95, edgecolor='#6c757d'),
                family='monospace', verticalalignment='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.45, left=0.05, right=0.95)
        
        print("✅ LOW POLY renderizado com sucesso!")
        print(f"   🔶 {len(triangles)} triângulos facetados")
        print(f"   ⚫ Arestas pretas destacadas")
        print(f"   🟨 Superfícies amarelas sólidas")
        print(f"   💎 Visual cristalino perfeito")
        
        plt.show()

def main():
    """Executa a criação do Pikachu Low Poly único"""
    print("🔶" + "="*70)
    print("                   PIKACHU LOW POLY")
    print("              Anatomia Exata e Detalhada")
    print("          Corpo Unificado • Rabo à Esquerda")
    print("        Olhos Curvos • Boca Amarela • Malha Triangular")
    print("="*74)
    
    # Criar sistema Low Poly
    pikachu = PikachuLowPoly()
    
    # Construir com coordenadas manuais
    pikachu.build_complete_low_poly_pikachu()
    
    # Renderizar ÚNICO Pikachu
    pikachu.render_single_pikachu()
    
    print("\n🏆 PIKACHU LOW POLY 3D MELHORADO CONCLUÍDO!")
    print("   🔶 Corpo hexagonal menos arredondado")
    print("   � Orelhas pontiagudas com pontas pretas")
    print("   ⚡ Rabo raio com espessura aumentada")
    print("   👀 Olhos grandes e mapeados (bem visíveis)")
    print("   🔴 Bochechas vermelhas destacadas")
    print("   😊 Boca amarela sorridente mapeada")
    print("   👃 Nariz pequeno triangular")
    print("   🤲 Mãos com 3 dedos característicos")
    print("   📊 Visualização 3D com grid espacial")
def main():
    """Executa a criação do Pikachu Low Poly MELHORADO"""
    print("🔶" + "="*75)
    print("                   PIKACHU LOW POLY 3D")
    print("              Modelo Melhorado e Ampliado")
    print("          Rabo Espesso • Características Mapeadas")
    print("        Visualização 3D • Grid Espacial • Mesh Triangular")
    print("="*78)
    
    # Criar sistema Low Poly MELHORADO
    pikachu = PikachuLowPoly()
    
    # Construir com coordenadas manuais otimizadas
    pikachu.build_complete_low_poly_pikachu()
    
    # Renderizar Pikachu GRANDE em 3D
    pikachu.render_single_pikachu()
    
    print("\n🏆 PIKACHU LOW POLY 3D MELHORADO CONCLUÍDO!")
    print("   🔶 Corpo hexagonal menos arredondado")
    print("   👂 Orelhas pontiagudas com pontas pretas")
    print("   ⚡ Rabo raio com espessura aumentada")
    print("   👀 Olhos grandes e mapeados (bem visíveis)")
    print("   🔴 Bochechas vermelhas destacadas")
    print("   😊 Boca amarela sorridente mapeada")
    print("   👃 Nariz pequeno triangular")
    print("   🤲 Mãos com 3 dedos característicos")
    print("   📊 Visualização 3D com grid espacial")
    print("   💎 Malha Low Poly triangular cristalina")
    print("   📏 Modelo ampliado para melhor detalhamento")

if __name__ == "__main__":
    main()
    print("   👂 Orelhas longas com pontas arredondadas")
    print("   ⚡ Rabo à esquerda (formato raio)")
    print("   � Olhos grandes com curva suave")
    print("   😊 Boca pequena amarela sorridente")
    print("   💎 Malha poligonal triangular cristalina")

if __name__ == "__main__":
    main()
