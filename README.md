# FinalProject – Setup & Usage (Windows + PyCharm)

## Requirements
- Windows 10/11
- Python 3.12 installed (recommended)
- PyCharm (Community/Professional)
- (Optional) NVIDIA GPU + CUDA drivers (for faster training)

---

## Installation (first time only)

### 1) Create a virtual environment
Open a terminal in the project folder and run:

```bash
py -3.12 -m venv .venv
```

### 2) Activate the environment
```bash
.\.venv\Scripts\activate
```

You should see `(.venv)` at the beginning of the terminal line.

### 3) Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Sanity check
```bash
python --version
python -c "import torch, torchvision, timm; print('OK')"
```

---

## Daily workflow (every time you open the project)
Activate the environment (no re-install needed):

```bash
.\.venv\Scripts\activate
```

Then work normally in PyCharm / run scripts.

---

## PyCharm Interpreter (important)
Make sure the project uses the virtualenv interpreter:

`<project_folder>\.venv\Scripts\python.exe`

In PyCharm:
- File → Settings → Project → Python Interpreter  
- Select the interpreter that points to `.venv\Scripts\python.exe`

---

## Adding a new package
If you install a new package, update `requirements.txt` so the other teammate can sync:

```bash
pip install <package_name>
pip freeze > requirements.txt
```

Then the other teammate runs:

```bash
pip install -r requirements.txt
```

---

## What to sync / what NOT to sync
### Sync to Drive / Git (share)
- `src/`, `configs/`, `main.py`, `README.md`
- `requirements.txt`
- (Optional) `runs/` and `artifacts/` if you want to share results/checkpoints

### Do NOT sync (local only)
- `.venv/` (virtual environment)
- `.idea/` (PyCharm local settings)

---

## Troubleshooting

### Terminal shows Python 3.8 (wrong interpreter)
Check where Python is coming from:

```bash
where python
python --version
```

Fix by activating the project venv:

```bash
.\.venv\Scripts\activate
```

### pip not found / packaging tools missing
```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
```
"# Object-Classification-in-Low-Resolution-THz-Imagery---" 
