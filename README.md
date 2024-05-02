# Python Template Repository

# Running this

For creating and regenerating environment file:
```
conda env create --file env.yaml
conda env export --no-builds | grep -v "^prefix: " > env.yaml
```