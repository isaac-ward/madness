# madness

## Reproducibility

We use Docker to run this project, which can be installed [here](https://docs.docker.com/engine/install/). See the ```./assets/reproducibility/``` folder for screenshots showing what the installation process should look like. Ensure that the Docker daemon is running, and then run the following commands **from the root folder of the project**:

```bash
# Build the Docker image
docker-compose build --no-cache
# Or you can remove the --no-cache flag if you want to cache the build and speed up future builds
docker-compose build
# Run a container of the built image
docker-compose up -d --no-deps
# Open a shell in that container
docker exec -it madness bash
# Execute the main script in that shell
python /mnt/src/main.py
# Exit from the shell/container when done
exit
# Close the container from the host machine
docker-compose down
```

This should produce a result in the ```logs``` directory.

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

