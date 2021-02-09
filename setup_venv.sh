virtualenv -p python3.8 .venv

source .venv/bin/activate

pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install hydra-core --upgrade
pip install hydra-submitit-launcher --upgrade
pip install -r requirements.txt
pip install -e .