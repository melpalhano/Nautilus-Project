@echo off
echo ⚡ PIKACHU MESHES VISUALIZADOR ⚡
echo ================================
echo.
echo 🎯 MESHES GERADOS ENCONTRADOS:
echo.

if exist "pikachu_mesh_suprema.obj" (
    echo ✅ pikachu_mesh_suprema.obj - 7.5 KB
    echo    🌊 Algoritmo Nautilus Supremo
    echo    🔗 Conecta TODOS os pontos do point cloud
    echo.
) else (
    echo ❌ pikachu_mesh_suprema.obj - NÃO ENCONTRADO
    echo.
)

if exist "pikachu_mesh_perfeita.obj" (
    echo ✅ pikachu_mesh_perfeita.obj - 65.4 KB  
    echo    💎 Malha de alta qualidade
    echo    🎯 Forma otimizada do Pikachu
    echo.
) else (
    echo ❌ pikachu_mesh_perfeita.obj - NÃO ENCONTRADO
    echo.
)

if exist "pikachu_mesh_nautilus_real.obj" (
    echo ✅ pikachu_mesh_nautilus_real.obj - 3.8 KB
    echo    🌊 Nautilus com forma real
    echo    ⚡ Aparência realística do Pikachu
    echo.
) else (
    echo ❌ pikachu_mesh_nautilus_real.obj - NÃO ENCONTRADO
    echo.
)

if exist "pikachu_mesh_convex_hull_otimizado.obj" (
    echo ✅ pikachu_mesh_convex_hull_otimizado.obj - 6.1 KB
    echo    🔺 ConvexHull otimizado
    echo    🌊 Geometria fechada garantida
    echo.
) else (
    echo ❌ pikachu_mesh_convex_hull_otimizado.obj - NÃO ENCONTRADO
    echo.
)

if exist "pikachu_mesh_delaunay_surface.obj" (
    echo ✅ pikachu_mesh_delaunay_surface.obj - 279.9 KB
    echo    🔷 Delaunay triangulação
    echo    📊 Máximo detalhe possível
    echo.
) else (
    echo ❌ pikachu_mesh_delaunay_surface.obj - NÃO ENCONTRADO
    echo.
)

echo 🎨 PARA VISUALIZAR OS MESHES:
echo.
echo 1. Abra o Windows 3D Viewer ou Paint 3D
echo 2. Importe qualquer arquivo .obj acima
echo 3. Use os controles de mouse para rotacionar
echo.
echo 🏆 RECOMENDAÇÃO: 
echo    pikachu_mesh_suprema.obj - Melhor resultado Nautilus
echo.
echo ⚡ Ou abra qualquer software de visualização 3D como:
echo    • Blender (gratuito)
echo    • MeshLab (gratuito) 
echo    • 3D Viewer do Windows
echo.

pause
