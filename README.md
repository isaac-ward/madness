# Python Template Repository

# Running this

We use [conda](https://docs.anaconda.com/free/miniconda/miniconda-install/) to manage the environment. 

For creating an environment from the environment file:
```
conda env create --file env.yaml
```

For updating the environment file after making changes:
```
conda env export --no-builds | grep -v "^prefix: " > env.yaml
```

Can then run any of the main files like so:
```
conda activate madness
python src/main.py
```

# Links

For 3D visualization:

- https://www.vpython.org/
- https://www.panda3d.org/features/
- https://github.com/fwilliams/point-cloud-utils
- https://towardsdatascience.com/python-libraries-for-mesh-and-point-cloud-visualization-part-1-daa2af36de30
- https://towardsdatascience.com/python-libraries-for-mesh-point-cloud-and-data-visualization-part-2-385f16188f0f