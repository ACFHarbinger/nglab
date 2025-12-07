# nglab
Nothing Gambles Like A Bot (NGLAB), a Multimodal Deep Reinforcement Learning bot to assist you in all your gambling (stock market) needs!

## Datasets
### Stock Market Data
[Stocks and ETFs from 1999 to 2020](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

### Newspaper Data
[BBC news articles from 2004 to 2005](http://mlg.ucd.ie/datasets/bbc.html?trk=article-ssr-frontend-pulse_little-text-block)

[Australia news headlines from 2003 t0 2021](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SYBGZL&trk=article-ssr-frontend-pulse_little-text-block)

[World politics news dataset](https://newsdata.io/files/datasets/world-politics-news)

[Global news dataset](https://www.kaggle.com/datasets/everydaycodings/global-news-dataset)

### Social Media Data
[Tweets](https://www.kaggle.com/datasets/bhavikjikadara/tweets-dataset/data)

## Setup Dependencies
You can choose to install this repository's dependencies using any of the following methods below.

### UV
To use the [UV Python package and project manager](https://github.com/astral-sh/uv) to setup the virtual environment, you just have to synchronize the project.
```bash
uv sync
```

Afterwards, you can initialize the virtual environment by running one of the following commands: 
- On the Linux CLI: `source .venv/bin/activate`
- On the Windows CMD: `.venv\Scripts\activate.bat`
- On the Windows PS: `.venv\Scripts\Activate.ps1`

After activating the virtual environment, you can list the installed packages in a similar manner to Conda by using Pip through UV:
```bash
uv pip list
```

Also, if you want to deactivate and/or delete the created virtual environment you can execute the following command(s).
```bash
deactivate
rm -rf .venv
```

#### UV Installation
To install UV, you simply need to execute the command `curl -LsSf https://astral.sh/uv/install.sh | sh` on the Linux CLI (or `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` on the Windows CMD|PS).

### Anaconda Environment
To setup the environment for the project using the [Anaconda distribution](https://www.anaconda.com/), you just need to run the following commands in the main directory:
```bash
conda env create --file env/environment.yml -y --name wsr
conda activate wsr
```

To list the installed packages (and their respective versions), just run the following command after activating the Conda environment:
```bash
conda list
```

and if you want to deactivate and/or delete the previously created Conda environment:
```bash
conda deactivate
conda remove -n wsr --all -y
```

#### Conda Installation
If you need to install conda beforehand, you just need to run the following commands (while replacing the variables for the values you want to use, which determine your Anaconda version):
```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-<year>.<month>-<version_id>-Linux-x86_64.sh
bash Anaconda3-<year>.<month>-<version_id>-Linux-x86_64.sh
```
For this project, we recommend you use Anaconda 3 with year=2024, month=10, version_id=1.

### Virtual Environment
To setup the virtual environment for the project using the Pip package installer and Python's venv module:
```bash
python3 -m venv env/.wsr
source env/.wsr/bin/activate
pip install -r env/requirements.txt
pip install -r env/pip_requirements.txt
```

After activating the virtual environment, you can list the installed packages in a similar manner to Conda by using Pip:
```bash
pip list
```

and if you want to deactivate and/or delete the created virtual environment:
```bash
deactivate
rm -rf env/.wsr
```

Note: to use this method, you already need to have the correct version (3.11) of Python 3 already installed in your system.

### Setup Scripts
You can also execute a script to completely setup your virtual environment using your preferred method. To do that, you simply need to execute the Linux command
```bash
bash scripts/setup_env.sh <selected_method>
```

or the following command on the Windows CMD:
```cmd
scripts\setup_env.bat <selected_method>
```

Note: the selected_method variable shoud be replaced with -> uv|conda|venv