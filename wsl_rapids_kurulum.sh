#!/bin/bash
# WSL2 (Ubuntu) içinde RAPIDS (cuML, cuDF) ortamını kurar
# Ubuntu terminalinde çalıştır:
#   bash /mnt/c/Users/Furkan/Desktop/losemi_mikrodizi/Kod/wsl_rapids_kurulum.sh
set -e

echo "== 1) GPU kontrolü =="
nvidia-smi

echo "== 2) Miniforge (conda) kurulumu =="
if [ ! -d "$HOME/miniforge3" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

echo "== 3) rapids ortamı: RAPIDS 26.10 + projede kullanılan paketler =="
if ! conda env list | grep -q "^rapids "; then
    conda create -y -n rapids -c rapidsai -c conda-forge -c nvidia \
        rapids=26.10 python=3.12 "cuda-version>=12.0,<=12.9" \
        scipy statsmodels matplotlib seaborn
fi
conda activate rapids

echo "== 4) Test =="
python -c "import cuml, cudf; print('cuML', cuml.__version__, '| cuDF', cudf.__version__)"
python -c "import cupy; print('GPU:', cupy.cuda.runtime.getDeviceProperties(0)['name'].decode())"
echo "Kurulum tamam. PyCharm yorumlayıcısı: $HOME/miniforge3/envs/rapids/bin/python"
