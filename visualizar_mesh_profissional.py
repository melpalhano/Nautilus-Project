#!/usr/bin/env python3
"""
🔥 VISUALIZADOR INTERATIVO PROFISSIONAL DA MESH PERFEITA
=========================================================

Visualização em tempo real da mesh perfeita com:
- Rotação 3D interativa
- Zoom e pan
- Múltiplos modos de visualização
- Análise de qualidade em tempo real
- Comparação lado a lado
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.widgets import Button, RadioButtons
import os

class VisualizadorMeshProfissional:
    """Visualizador profissional interativo da mesh"""
    
    def __init__(self):
        self.mesh = None
        self.fig = None
        self.ax_3d = None
        self.modo_visualizacao = 'points'
        self.cor_atual = 'viridis'
        
    def carregar_mesh(self, arquivo='pikachu_mesh_perfeita.obj'):
        """Carrega a mesh perfeita"""
        try:
            self.mesh = trimesh.load(arquivo)
            print(f"✅ Mesh carregada: {arquivo}")
            print(f"   🔺 {len(self.mesh.vertices):,} vértices")
            print(f"   🔺 {len(self.mesh.faces):,} faces")
            print(f"   🌊 Watertight: {self.mesh.is_watertight}")
            return True
        except Exception as e:
            print(f"❌ Erro carregando mesh: {e}")
            return False
    
    def configurar_interface(self):
        """Configura interface profissional"""
        # Figura principal
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('🔥 PIKACHU MESH PERFEITA - VISUALIZADOR PROFISSIONAL 🔥', 
                         fontsize=16, fontweight='bold')
        
        # Layout: 3D principal + painéis laterais
        self.ax_3d = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3, projection='3d')
        
        # Painel de informações
        self.ax_info = plt.subplot2grid((3, 4), (0, 3))
        self.ax_info.axis('off')
        
        # Painel de controles
        self.ax_controles = plt.subplot2grid((3, 4), (1, 3))
        self.ax_controles.axis('off')
        
        # Painel de estatísticas
        self.ax_stats = plt.subplot2grid((3, 4), (2, 3))
        self.ax_stats.axis('off')
        
        # Configura visualização 3D
        self.configurar_3d()
        
        # Adiciona controles
        self.adicionar_controles()
        
        # Atualiza informações
        self.atualizar_paineis()
    
    def configurar_3d(self):
        """Configura o plot 3D principal"""
        # Remove eixos para visualização limpa
        self.ax_3d.set_xlabel('X', fontweight='bold')
        self.ax_3d.set_ylabel('Y', fontweight='bold')
        self.ax_3d.set_zlabel('Z', fontweight='bold')
        
        # Configuração visual
        self.ax_3d.grid(True, alpha=0.3)
        self.ax_3d.set_facecolor('black')
        
        # Vista inicial
        self.ax_3d.view_init(elev=30, azim=45)
    
    def adicionar_controles(self):
        """Adiciona controles interativos"""
        # Radio buttons para modo de visualização
        self.radio_ax = plt.axes([0.78, 0.4, 0.15, 0.15])
        self.radio = RadioButtons(self.radio_ax, 
                                 ('Points', 'Wireframe', 'Surface', 'Normals'),
                                 active=0)
        self.radio.on_clicked(self.mudar_modo)
        
        # Botões de vista
        self.btn_front = Button(plt.axes([0.78, 0.25, 0.06, 0.04]), 'Front')
        self.btn_front.on_clicked(lambda x: self.mudar_vista(0, 0))
        
        self.btn_side = Button(plt.axes([0.85, 0.25, 0.06, 0.04]), 'Side')
        self.btn_side.on_clicked(lambda x: self.mudar_vista(0, 90))
        
        self.btn_top = Button(plt.axes([0.78, 0.20, 0.06, 0.04]), 'Top')
        self.btn_top.on_clicked(lambda x: self.mudar_vista(90, 0))
        
        self.btn_iso = Button(plt.axes([0.85, 0.20, 0.06, 0.04]), 'Iso')
        self.btn_iso.on_clicked(lambda x: self.mudar_vista(30, 45))
        
        # Botão de reset
        self.btn_reset = Button(plt.axes([0.78, 0.15, 0.13, 0.04]), 'Reset View')
        self.btn_reset.on_clicked(self.reset_view)
    
    def mudar_modo(self, modo):
        """Muda modo de visualização"""
        modo_map = {
            'Points': 'points',
            'Wireframe': 'wireframe', 
            'Surface': 'surface',
            'Normals': 'normals'
        }
        
        self.modo_visualizacao = modo_map[modo]
        self.atualizar_visualizacao()
    
    def mudar_vista(self, elev, azim):
        """Muda vista 3D"""
        self.ax_3d.view_init(elev=elev, azim=azim)
        plt.draw()
    
    def reset_view(self, event):
        """Reset da visualização"""
        self.ax_3d.clear()
        self.configurar_3d()
        self.atualizar_visualizacao()
        plt.draw()
    
    def atualizar_visualizacao(self):
        """Atualiza visualização 3D baseada no modo"""
        if self.mesh is None:
            return
        
        # Limpa plot
        self.ax_3d.clear()
        self.configurar_3d()
        
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        if self.modo_visualizacao == 'points':
            # Modo pontos coloridos por altura
            scatter = self.ax_3d.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                       c=vertices[:, 2], cmap=self.cor_atual, 
                                       s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=self.ax_3d, shrink=0.8, label='Altura (Z)')
            
        elif self.modo_visualizacao == 'wireframe':
            # Modo wireframe
            for i in range(0, len(faces), max(1, len(faces)//200)):
                face = faces[i]
                triangle = vertices[face]
                triangle_closed = np.vstack([triangle, triangle[0]])
                self.ax_3d.plot(triangle_closed[:, 0], triangle_closed[:, 1], triangle_closed[:, 2],
                               'cyan', alpha=0.7, linewidth=1)
            
        elif self.modo_visualizacao == 'surface':
            # Modo surface com faces
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            # Amostra de faces para performance
            face_sample = faces[::max(1, len(faces)//500)]
            triangles = vertices[face_sample]
            
            # Cria coleção de polígonos
            poly3d = Poly3DCollection(triangles, alpha=0.7, facecolor='cyan', 
                                    edgecolor='blue', linewidth=0.5)
            self.ax_3d.add_collection3d(poly3d)
            
        elif self.modo_visualizacao == 'normals':
            # Modo normais das faces
            if hasattr(self.mesh, 'face_normals'):
                face_centers = self.mesh.triangles_center
                normals = self.mesh.face_normals
                
                # Amostra para performance
                sample_idx = np.arange(0, len(face_centers), max(1, len(face_centers)//100))
                centers = face_centers[sample_idx]
                norms = normals[sample_idx] * 0.1  # Escala das normais
                
                # Desenha normais como setas
                for i in range(len(centers)):
                    start = centers[i]
                    end = start + norms[i]
                    self.ax_3d.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                                   'red', linewidth=2, alpha=0.8)
                
                # Pontos dos centros
                self.ax_3d.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                                 c='yellow', s=30, alpha=0.9)
        
        # Ajusta limites automaticamente
        self.ax_3d.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        
        # Título do modo atual
        self.ax_3d.set_title(f'🎯 Modo: {self.modo_visualizacao.title()}', 
                           fontweight='bold', fontsize=12)
        
        plt.draw()
    
    def atualizar_paineis(self):
        """Atualiza painéis de informação"""
        if self.mesh is None:
            return
        
        # Painel de informações básicas
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = f"""
🔥 MESH PERFEITA 🔥

📊 ESTATÍSTICAS:
🔺 Vértices: {len(self.mesh.vertices):,}
🔺 Faces: {len(self.mesh.faces):,}
🔺 Arestas: {len(self.mesh.edges):,}

🏗️ QUALIDADE:
📐 Área: {self.mesh.area:.4f}
🌊 Watertight: {'✅' if self.mesh.is_watertight else '❌'}
🔄 Winding: {'✅' if self.mesh.is_winding_consistent else '❌'}
"""
        
        if self.mesh.is_watertight:
            info_text += f"📦 Volume: {self.mesh.volume:.4f}\n"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        # Painel de controles
        self.ax_controles.clear()
        self.ax_controles.axis('off')
        
        controles_text = """
🎮 CONTROLES:

🖱️ Mouse:
• Arrastar: Rotacionar
• Scroll: Zoom
• Shift+Arrastar: Pan

📊 Modos:
• Points: Pontos 3D
• Wireframe: Malha
• Surface: Superfície
• Normals: Normais

🎯 Vistas Rápidas:
• Front/Side/Top/Iso
"""
        
        self.ax_controles.text(0.05, 0.95, controles_text, transform=self.ax_controles.transAxes,
                              fontsize=8, verticalalignment='top', fontfamily='monospace')
        
        # Painel de estatísticas avançadas
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Bounding box
        bbox = self.mesh.bounds
        
        stats_text = f"""
📦 DIMENSÕES:
X: {bbox[1][0]-bbox[0][0]:.3f}
Y: {bbox[1][1]-bbox[0][1]:.3f}
Z: {bbox[1][2]-bbox[0][2]:.3f}

⚖️ CENTRO: 
({self.mesh.center_mass[0]:.2f}, 
 {self.mesh.center_mass[1]:.2f}, 
 {self.mesh.center_mass[2]:.2f})

🎯 QUALIDADE:
{'🏆 EXCELENTE!' if self.mesh.is_watertight else '⚠️ Melhorar'}
"""
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=8, verticalalignment='top', fontfamily='monospace')
    
    def iniciar_visualizacao(self):
        """Inicia visualização interativa"""
        print("🚀 INICIANDO VISUALIZADOR PROFISSIONAL...")
        
        if not self.carregar_mesh():
            return
        
        # Configura interface
        self.configurar_interface()
        
        # Primeira visualização
        self.atualizar_visualizacao()
        
        # Instruções
        print("\n🎮 CONTROLES DISPONÍVEIS:")
        print("   🖱️ Mouse: Arrastar para rotacionar, scroll para zoom")
        print("   📊 Radio buttons: Mudar modo de visualização")
        print("   🎯 Botões: Vistas rápidas (Front, Side, Top, Iso)")
        print("   🔄 Reset View: Volta à vista inicial")
        print("\n✨ APROVEITE A VISUALIZAÇÃO PROFISSIONAL!")
        
        # Mostra interface
        plt.tight_layout()
        plt.show()

def main():
    """Função principal"""
    print("🔥 VISUALIZADOR PROFISSIONAL DA MESH PERFEITA")
    print("="*60)
    print("🎯 VISUALIZAÇÃO INTERATIVA DE QUALIDADE MÁXIMA!")
    print("🎯 MÚLTIPLOS MODOS E CONTROLES AVANÇADOS!")
    print("="*60)
    
    # Verifica se a mesh existe
    if not os.path.exists('pikachu_mesh_perfeita.obj'):
        print("❌ Mesh perfeita não encontrada!")
        print("   Execute primeiro: python pikachu_mesh_perfeita.py")
        return
    
    # Cria e inicia visualizador
    visualizador = VisualizadorMeshProfissional()
    visualizador.iniciar_visualizacao()

if __name__ == "__main__":
    main()
