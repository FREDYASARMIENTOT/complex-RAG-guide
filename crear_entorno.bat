@echo off
echo ===================================================
echo Creando Entorno Conda para Complex RAG Guide
echo ===================================================
echo.

REM Verificar si conda esta instalado
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Conda no esta instalado o no se encuentra en el PATH.
    echo Por favor, instala Anaconda o Miniconda y asegurate de agregarlo al PATH.
    pause
    exit /b 1
)

echo [1/4] Creando el entorno conda 'complex_rag' con Python 3.10...
call conda create -n complex_rag python=3.10 -y

if %errorlevel% neq 0 (
    echo [ERROR] Fallo al crear el entorno de Conda.
    pause
    exit /b 1
)

echo.
echo [2/4] Registrando el kernel de Jupyter...
call conda run -n complex_rag pip install ipykernel
call conda run -n complex_rag python -m ipykernel install --user --name complex_rag --display-name "Python (Complex RAG)"

echo.
echo [3/4] Instalando dependencias desde requirements.txt...
call conda run -n complex_rag pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Fallo al instalar las dependencias de requirements.txt.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo [4/4] ¡Entorno conda 'complex_rag' creado con exito!
echo ===================================================
echo.
echo Siguientes pasos:
echo 1. Abre el notebook 'RAG_pipeline.ipynb' en tu IDE (VS Code, Jupyter, etc.).
echo 2. Selecciona el Kernel de Jupyter llamado: "Python (Complex RAG)".
echo 3. ¡Listo! Ya puedes ejecutar las celdas sin problemas de modulos.
echo.
pause
