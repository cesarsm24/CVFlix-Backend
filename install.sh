#!/bin/bash
# ==============================================================================
# CVFlix - Instalador Universal
#
# Soporta: macOS (Intel/ARM), Linux (Ubuntu, Debian, Fedora, Arch), WSL
# Estrategia: Conda para dlib (wheels), pip para el resto
# Tiempo: 3-5 minutos
#
# Uso:
#   bash install.sh              # Instalar y ejecutar
#   bash install.sh --clean      # Limpiar todo
#   bash install.sh --check      # Verificar sistema
# ==============================================================================

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# ==================== CONFIGURACIÓN ====================
ENV_NAME="cvflix"
PYTHON_VERSION="3.11"
TMDB_API_KEY="2d51820b0b76e3ea8a7d2862af21839a"

# Variables globales
OS=""
ARCH=""
DISTRO=""
PACKAGE_MANAGER=""
CONDA_INSTALLED_BY_SCRIPT=false

# ==============================================================================
# FUNCIONES DE UI
# ==============================================================================

print_header() {
    echo -e "\n${BLUE}╔════════════════════════════════════════╗${NC}"
    printf "${BLUE}║  %-38s${BLUE}║${NC}\n" "$1"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"
}

print_step() { echo -e "${CYAN}▶ $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${MAGENTA}ℹ️  $1${NC}"; }

# ==============================================================================
# DETECCIÓN DE SISTEMA
# ==============================================================================

detect_system() {
    print_header "🔍 Detectando Sistema"

    # Sistema Operativo
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macOS $(sw_vers -productVersion)"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"

        if [ -f /etc/os-release ]; then
            . /etc/os-release
            DISTRO="$NAME $VERSION_ID"
        else
            DISTRO="Linux"
        fi

        # Detectar WSL
        if grep -qi microsoft /proc/version 2>/dev/null; then
            OS="wsl"
            DISTRO="$DISTRO (WSL)"
        fi
    else
        print_error "Sistema no soportado: $OSTYPE"
        exit 1
    fi

    ARCH=$(uname -m)
    detect_package_manager

    print_info "Sistema: $DISTRO"
    print_info "Arquitectura: $ARCH"
    print_info "Gestor de paquetes: $PACKAGE_MANAGER"
    echo ""
}

detect_package_manager() {
    if [[ "$OS" == "macos" ]]; then
        PACKAGE_MANAGER=$(command -v brew &> /dev/null && echo "homebrew" || echo "none")
    elif [[ "$OS" == "linux" ]] || [[ "$OS" == "wsl" ]]; then
        if command -v apt-get &> /dev/null; then
            PACKAGE_MANAGER="apt"
        elif command -v dnf &> /dev/null; then
            PACKAGE_MANAGER="dnf"
        elif command -v pacman &> /dev/null; then
            PACKAGE_MANAGER="pacman"
        else
            PACKAGE_MANAGER="none"
        fi
    fi
}

# ==============================================================================
# PREREQUISITOS
# ==============================================================================

install_prerequisites() {
    print_header "📦 Prerequisitos"

    case $OS in
        macos)
            # Homebrew
            if ! command -v brew &> /dev/null; then
                print_step "Instalando Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

                if [[ "$ARCH" == "arm64" ]]; then
                    eval "$(/opt/homebrew/bin/brew shellenv)"
                    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
                fi

                print_success "Homebrew instalado"
            else
                print_success "Homebrew ya instalado"
            fi

            # Xcode Command Line Tools
            if ! xcode-select -p &> /dev/null; then
                print_step "Instalando Xcode Command Line Tools..."
                xcode-select --install
                print_warning "Completa la instalación y ejecuta el script de nuevo"
                exit 0
            else
                print_success "Xcode Command Line Tools instalado"
            fi
            ;;

        linux|wsl)
            case $PACKAGE_MANAGER in
                apt)
                    sudo apt update -qq
                    sudo apt install -y wget curl git -qq
                    ;;
                dnf)
                    sudo dnf install -y wget curl git -q
                    ;;
                pacman)
                    sudo pacman -Sy --noconfirm wget curl git
                    ;;
            esac
            print_success "Herramientas básicas instaladas"
            ;;
    esac
}

# ==============================================================================
# CONDA
# ==============================================================================

install_conda() {
    print_header "🐍 Configurando Conda"

    if command -v conda &> /dev/null; then
        print_success "Conda ya instalado: $(conda --version)"
        eval "$(conda shell.bash hook)"

        # Aceptar ToS si es necesario
        accept_conda_tos
        return 0
    fi

    print_step "Instalando Miniconda..."

    case $OS in
        macos)
            brew install --cask miniconda

            if [[ "$ARCH" == "arm64" ]]; then
                CONDA_PATH="/opt/homebrew/Caskroom/miniconda/base"
            else
                CONDA_PATH="/usr/local/Caskroom/miniconda/base"
            fi

            ${CONDA_PATH}/bin/conda init bash 2>/dev/null || true
            ${CONDA_PATH}/bin/conda init zsh 2>/dev/null || true
            eval "$(${CONDA_PATH}/bin/conda shell.bash hook)"
            ;;

        linux|wsl)
            case $ARCH in
                x86_64)
                    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
                    ;;
                aarch64|arm64)
                    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
                    ;;
                *)
                    print_error "Arquitectura no soportada: $ARCH"
                    exit 1
                    ;;
            esac

            wget -q --show-progress "$URL" -O /tmp/miniconda.sh
            bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
            rm /tmp/miniconda.sh

            "$HOME/miniconda3/bin/conda" init bash
            eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
            ;;
    esac

    CONDA_INSTALLED_BY_SCRIPT=true
    touch "$HOME/.cvflix_conda_installed"

    # Aceptar ToS después de instalar
    accept_conda_tos

    print_success "Miniconda instalado"
}

# Función para aceptar términos de servicio de Conda
accept_conda_tos() {
    print_step "Verificando términos de servicio de Conda..."

    # Intentar aceptar los ToS automáticamente
    conda config --set allow_conda_downgrades true 2>/dev/null || true

    # Aceptar ToS para los canales principales
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

    # También aceptar para conda-forge (por si acaso)
    conda tos accept --override-channels --channel conda-forge 2>/dev/null || true

    print_success "Términos de servicio aceptados"
}

# ==============================================================================
# ENTORNO PYTHON
# ==============================================================================

create_environment() {
    print_header "🐍 Entorno Python"

    # Eliminar si existe
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "Entorno '$ENV_NAME' existe - recreando..."
        conda deactivate 2>/dev/null || true
        conda env remove -n ${ENV_NAME} -y -q
    fi

    # Crear
    print_step "Creando entorno Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y -q

    # Activar
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}

    print_success "Entorno '$ENV_NAME' creado"
    print_info "Python: $(python --version)"
}

# ==============================================================================
# DEPENDENCIAS (ESTRATEGIA ÓPTIMA)
# ==============================================================================

install_dependencies() {
    print_header "📥 Instalando Dependencias"

    if [ ! -f "requirements.txt" ]; then
        print_error "No se encontró requirements.txt"
        exit 1
    fi

    echo -e "${CYAN}Estrategia:${NC}"
    echo "  1️⃣  Conda → dlib (wheel precompilado)"
    echo "  2️⃣  pip → resto (wheels oficiales)"
    echo ""
    print_warning "Tiempo estimado: 3-5 minutos"
    echo ""

    # Actualizar pip
    print_step "Actualizando pip..."
    pip install --upgrade pip setuptools wheel -q

    # ========================================
    # FASE 1: dlib via conda
    # ========================================
    print_step "FASE 1/3: dlib (wheel conda-forge)..."

    if python -c "import dlib" 2>/dev/null; then
        local version=$(python -c "import dlib; print(dlib.__version__)" 2>/dev/null || echo "OK")
        print_success "dlib ya instalado: ${version}"
    else
        if conda install -c conda-forge dlib -y -q 2>&1 | tail -n 1; then
            print_success "dlib instalado (~30 seg)"
        else
            print_warning "conda falló, intentando compilar..."
            print_warning "Esto puede tardar 15-20 minutos..."

            timeout 1800 pip install dlib --no-cache-dir 2>&1 | tail -n 5 || {
                print_error "dlib falló - continuando sin face recognition"
            }
        fi
    fi

    # ========================================
    # FASE 2: Dependencias principales
    # ========================================
    print_step "FASE 2/3: Dependencias principales..."

    # Crear requirements temporal SIN face-recognition
    grep -v "face-recognition" requirements.txt > /tmp/requirements_temp.txt

    if pip install -r /tmp/requirements_temp.txt -q 2>&1 | tail -n 3; then
        print_success "Dependencias instaladas (~2 min)"
    else
        print_warning "Algunas dependencias fallaron"
    fi

    rm -f /tmp/requirements_temp.txt

    # ========================================
    # FASE 3: Face recognition
    # ========================================
    print_step "FASE 3/3: Face recognition..."

    if python -c "import dlib" 2>/dev/null; then
        if pip install face-recognition face-recognition-models -q; then
            print_success "Face recognition instalado"
        else
            print_warning "Face recognition falló"
        fi
    else
        print_warning "dlib no disponible - omitiendo face recognition"
    fi

    # ========================================
    # FASE 4: TensorFlow (específico por sistema)
    # ========================================
    print_step "FASE 4/4: TensorFlow..."

    if python -c "import tensorflow" 2>/dev/null; then
        local version=$(python -c "import tensorflow; print(tensorflow.__version__)" 2>/dev/null)
        print_success "TensorFlow ya instalado: ${version}"
    else
        install_tensorflow
    fi

    echo ""
    print_success "Instalación completada"
}

install_tensorflow() {
    case $OS in
        macos)
            if [[ "$ARCH" == "arm64" ]]; then
                # Apple Silicon: tensorflow-macos
                print_info "Apple Silicon detectado - instalando tensorflow-macos..."
                if pip install tensorflow-macos tensorflow-metal -q 2>&1 | tail -n 3; then
                    print_success "TensorFlow (Apple Silicon) instalado"
                else
                    print_warning "tensorflow-macos falló, intentando versión estándar..."
                    pip install tensorflow -q 2>&1 | tail -n 3 || print_warning "TensorFlow no disponible"
                fi
            else
                # Intel Mac
                if pip install tensorflow -q 2>&1 | tail -n 3; then
                    print_success "TensorFlow instalado"
                else
                    print_warning "TensorFlow falló"
                fi
            fi
            ;;

        linux|wsl)
            # Linux: versión estándar
            if pip install tensorflow -q 2>&1 | tail -n 3; then
                print_success "TensorFlow instalado"
            else
                print_warning "TensorFlow falló, intentando versión CPU-only..."
                pip install tensorflow-cpu -q 2>&1 | tail -n 3 || print_warning "TensorFlow no disponible"
            fi
            ;;
    esac
}

# ==============================================================================
# VERIFICACIÓN
# ==============================================================================

verify_installation() {
    print_header "✓ Verificando Instalación"

    check_package() {
        if python -c "import $1" 2>/dev/null; then
            local version=$(python -c "import $1; print(getattr($1, '__version__', 'OK'))" 2>/dev/null)
            print_success "$2: ${version}"
            return 0
        else
            print_error "$2: FALTA"
            return 1
        fi
    }

    echo "Paquetes principales:"
    check_package "fastapi" "FastAPI"
    check_package "uvicorn" "Uvicorn"
    check_package "cv2" "OpenCV"
    check_package "numpy" "NumPy"

    echo ""
    echo "Face Recognition:"
    check_package "dlib" "dlib"
    check_package "face_recognition" "face-recognition"

    echo ""
    echo "Machine Learning:"
    check_package "tensorflow" "TensorFlow"
    check_package "sklearn" "scikit-learn"
}

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

configure_app() {
    print_header "⚙️  Configuración"

    # Crear .env
    if [ ! -f ".env" ]; then
        if [ -n "$TMDB_API_KEY" ] && [ "$TMDB_API_KEY" != "TU_CLAVE_TMDB_AQUI" ]; then
            echo "TMDB_API_KEY=${TMDB_API_KEY}" > .env
            print_success "Archivo .env creado"
        else
            cp .env.example .env 2>/dev/null || echo "# TMDB_API_KEY=tu_clave_aqui" > .env
            print_warning ".env creado - configura TMDB_API_KEY"
        fi
    else
        print_success "Archivo .env ya existe"
    fi

    # Scripts de ayuda
    create_helper_scripts
}

create_helper_scripts() {
    cat > start.sh << 'EOF'
#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cvflix
echo "🚀 Iniciando CVFlix..."
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
EOF
    chmod +x start.sh

    cat > activate.sh << 'EOF'
#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cvflix
echo "🎬 Entorno CVFlix activado"
EOF
    chmod +x activate.sh

    print_success "Scripts creados: start.sh, activate.sh"
}

# ==============================================================================
# EJECUTAR
# ==============================================================================

run_app() {
    print_header "🚀 Iniciando CVFlix"

    echo -e "${GREEN}Servidor: http://127.0.0.1:8000${NC}"
    echo -e "${GREEN}Docs: http://127.0.0.1:8000/docs${NC}"
    echo ""
    echo -e "${YELLOW}Presiona Ctrl+C para detener${NC}\n"

    trap 'echo -e "\n${YELLOW}Servidor detenido${NC}"; show_cleanup_menu' INT

    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
}

# ==============================================================================
# LIMPIEZA
# ==============================================================================

show_cleanup_menu() {
    echo ""
    print_header "🧹 Opciones"

    echo "  1) Mantener todo"
    echo "  2) Eliminar entorno (~2 GB)"
    echo "  3) Eliminar TODO (~3 GB)"
    echo ""
    read -p "Selecciona (1-3): " REPLY
    echo

    case $REPLY in
        2)
            conda deactivate 2>/dev/null || true
            conda env remove -n ${ENV_NAME} -y
            print_success "Entorno eliminado"
            ;;
        3)
            cleanup_all
            ;;
        *)
            print_success "Todo mantenido"
            ;;
    esac
}

cleanup_all() {
    print_header "🧹 Limpieza Completa"

    # Entorno
    if command -v conda &> /dev/null && conda env list | grep -q "^${ENV_NAME} "; then
        conda deactivate 2>/dev/null || true
        conda env remove -n ${ENV_NAME} -y
        print_success "Entorno eliminado"
    fi

    # Conda si fue instalado por script
    if [ -f "$HOME/.cvflix_conda_installed" ]; then
        case $OS in
            macos)
                brew uninstall --cask miniconda 2>/dev/null || true
                ;;
            linux|wsl)
                rm -rf "$HOME/miniconda3"
                ;;
        esac
        rm -f "$HOME/.cvflix_conda_installed"
        print_success "Conda eliminado"
    fi

    # Archivos
    rm -f .env start.sh activate.sh

    print_success "Limpieza completada"
}

check_system_only() {
    detect_system
    print_header "📋 Resumen"
    echo -e "${CYAN}Sistema:${NC} $DISTRO"
    echo -e "${CYAN}Arquitectura:${NC} $ARCH"
    echo -e "${CYAN}Conda:${NC} $(command -v conda &> /dev/null && echo "✅ Instalado" || echo "❌ No instalado")"
}

# ==============================================================================
# MAIN
# ==============================================================================

main() {
    case "$1" in
        --clean)
            detect_system
            cleanup_all
            exit 0
            ;;
        --check)
            check_system_only
            exit 0
            ;;
    esac

    print_header "🎬 CVFlix - Instalación"

    echo -e "${CYAN}Este script:${NC}"
    echo "  ✅ Detecta tu sistema automáticamente"
    echo "  ✅ Instala Conda (si no lo tienes)"
    echo "  ✅ Crea entorno Python 3.11"
    echo "  ✅ Instala dlib con wheels (rápido)"
    echo "  ✅ Instala todas las dependencias"
    echo "  ✅ Configura y ejecuta CVFlix"
    echo ""
    echo -e "${YELLOW}Tiempo estimado: 5-6 minutos${NC}"
    echo ""
    read -p "¿Continuar? (s/n): " REPLY
    echo
    [[ ! $REPLY =~ ^[Ss]$ ]] && exit 0

    detect_system
    install_prerequisites
    install_conda
    create_environment
    install_dependencies
    verify_installation
    configure_app
s
    print_header "✅ Instalación Completada"

    echo -e "${GREEN}Todo listo para ejecutar CVFlix${NC}\n"
    read -p "¿Iniciar servidor ahora? (s/n): " REPLY
    echo

    if [[ $REPLY =~ ^[Ss]$ ]]; then
        run_app
        show_cleanup_menu
    else
        print_info "Para iniciar después:"
        echo "  ./start.sh"
        echo ""
        echo "O manualmente:"
        echo "  conda activate cvflix"
        echo "  uvicorn app.main:app --reload"
    fi
}

main "$@"