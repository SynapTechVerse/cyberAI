python.exe -m pip install --upgrade pip
py -3.10  -m venv c-ai
.\c-ai\Scripts\activate

pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
