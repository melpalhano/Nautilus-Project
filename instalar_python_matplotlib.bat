@echo off
echo ========================================
echo INSTALADOR COMPLETO PYTHON + MATPLOTLIB
echo Para Visualizar Meshes do Pikachu
echo ========================================
echo.

echo [1/4] Instalando Python via Microsoft Store...
start ms-windows-store://pdp/?productid=9NRWMJP3717K
echo    ^^ Uma janela da Microsoft Store deve ter aberto
echo    ^^ Clique em INSTALAR e aguarde a instalacao
echo.
pause

echo [2/4] Aguardando instalacao do Python...
echo Pressione qualquer tecla DEPOIS que o Python estiver instalado
pause

echo [3/4] Testando instalacao do Python...
python --version
if %errorlevel% neq 0 (
    echo ERRO: Python nao foi instalado corretamente
    echo Tente novamente ou instale manualmente
    pause
    exit /b 1
)

echo [4/4] Instalando bibliotecas necessarias...
echo Instalando matplotlib...
python -m pip install matplotlib

echo Instalando numpy...  
python -m pip install numpy

echo.
echo ========================================
echo INSTALACAO CONCLUIDA!
echo ========================================
echo.
echo Para visualizar suas meshes, execute:
echo    python visualizador_matplotlib_pikachu.py
echo.
echo Ou para visualizar a mesh suprema diretamente:
echo    python visualizador_matplotlib_pikachu.py pikachu_mesh_suprema.obj
echo.
pause
