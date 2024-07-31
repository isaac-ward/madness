# madness

## Reproducibility

- Users should create and fill out a ```env/.env``` file; the file ```env/.env.example``` is provided as a template.
- Install Docker [here](https://docs.docker.com/engine/install/).
- **Ensure that the Docker daemon is running**.
- Run the following commands **from the root folder of the project**. This should handle all dependencies.

```bash
# Build the Docker image
docker-compose --progress plain build --no-cache
# Or you can remove the --no-cache flag to use cached builds
# (this speeds up future builds, but may cause issues if the Dockerfile is edited)
docker-compose --progress plain build
# Run a container of the built image
docker-compose up -d --no-deps
# Open a shell in that container
docker exec -it madness bash
# Execute any main script in that shell
python /mnt/src/main_mppi.py
python /mnt/src/main_train.py
python /mnt/src/main_make_data.py
bash mnt/src/bash/make_data.sh 
# Exit from the shell/container when done
exit
# Close the container from the host machine
docker-compose down
```

This should produce a result in the ```logs``` directory. See the ```./docs/reproducibility/``` folder for screenshots showing what the installation process should look like.

## Links

For 3D visualization:

- https://www.vpython.org/
- https://www.panda3d.org/features/
- https://github.com/fwilliams/point-cloud-utils
- https://towardsdatascience.com/python-libraries-for-mesh-and-point-cloud-visualization-part-1-daa2af36de30
- https://towardsdatascience.com/python-libraries-for-mesh-point-cloud-and-data-visualization-part-2-385f16188f0f

## Handy commands

To remove all logs (minus the gitkeep file)
```
rm -rf logs/*
```

