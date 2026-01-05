# AI.Accelerate Bootcamp — Starter Code

This repository contains two Python files that students will use to start their bootcamp projects. Files are:
- `datasets/`: Dataset directory for each of the three email-helper actions students are expected to support in their apps
- `app.py`: Starter Streamlit app

Follow the instructions below to set up your environment, create a virtual environment, and install all required dependencies.

---

## Getting Started

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### **2. Create a Virtual Environment**

> Recommended: Python 3.9+

A virtual environment keeps your project isolated from your system Python packages.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```
When activated, your terminal prompt should display (.venv).

In case this doesn't work for you, you can try creating a conda environment and install the packages there as follows:

```bash
conda create -n app python=3.11
conda activate app
conda install pyarrow pandas numpy
pip install -r requirements.txt --no-deps
```

### **3. Install Required Dependencies**

With the virtual environment activated, install the project dependencies:

```bash
pip install -r requirements.txt
```

### **3a. Update Dependencies**

If you install new packages while working on the project:

```bash
pip install <package-name>
pip freeze > requirements.txt
```
This regenerates requirements.txt so others can install the same environment.

### **4. Start the app**
To run the app on Streamlit, run:
```bash
streamlit run app.py
```
It will open a localhost on your browser. Use this to preview all the changes to your app during the bootcamp.

Welcome to AI.Accelerate!
