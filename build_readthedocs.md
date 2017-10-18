*Steps to use `.ipynb` to directly build readthedocs*

`pip install sphinx`
`pip install nbconvert`
`pip install latex`
`pip install nbsphinx`
`conda install pandoc`

Then you can use the followed to build and preview locally
```python3 -m sphinx 'your_source_dir' 'your_build_dir'```

A lot more functions of `nbconvert` can be found here (http://nbconvert.readthedocs.io/en/latest/index.html)

For setting up readthedocs:

1. create a `readthedocs.yml` in the master repo dir:

```
# Read The Docs config
# - http://docs.readthedocs.io/en/latest/yaml-config.html

# python version
python:
   version: 3.5
   pip_install: True

# build a PDF
formats:
  - none

# use a conda environment file
# - https://conda.io/docs/using/envs.html#share-an-environment
conda:
   file: documentation/readthedocs-environment.yml
```

 2. create a `readthedocs-environment.yml` in your documentation dir:
 
```
name: Your-Project-Name
channels:
  - conda-forge
dependencies:
  - python==3.5
  - pandoc
  - nbformat
  - jupyter_client
  - ipython
  - nbconvert
  - sphinx>=1.5.1
  - ipykernel
  - sphinx_rtd_theme
  - pip:
  - nbsphinx
```

 3. modify the conf.py file in your documentation source dir:

 ```
 extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.githubpages',
              'nbsphinx',
              'sphinx.ext.mathjax',
              'IPython.sphinxext.ipython_console_highlighting']
source_suffix = ['.rst', '.ipynb']
```

