# madness

## Reproducibility
### Installation and Setup
- Users should create and fill out a ```env/.env``` file; the file ```env/.env.example``` is provided as a template. This requires creating an account on https://wandb.ai/.
- Install Docker [here](https://docs.docker.com/engine/install/).
- **Ensure that the Docker daemon is running**. Edit the Docker Desktop settings under *Docker Engine* to allow for the nvidia runtime.
- Open Visual Studio Code at the project root
- Install the 'Dev Containers' extension
### Build and Open Contianer
- Use the keyboard command <kbd>Ctrl + Shift + P</kbd> (Windows) or <kbd>⌘ + ⇧ + P</kbd> (MacOS) to open the Command Palette.
- Type the command 'Dev Containers: Reopen in Container' and hit <kbd>Enter</kbd> (Windodws) or <kbd> return ⏎ </kbd> (MacOS)
- When prompted with a drop down menu, select "basic" for no gpu acceleration and "gpu" for gpu acceleration

### Unit Test
- Once you are in the container, open a terminal
- Run the following commands to perform a basic test of the MPPI guidance and control architecture

```bash
# Open a shell inside that container (multiple shells can be opened in one container)
docker exec -it madness bash
# Execute any main script in that shell
python /src/main_mppi.py
python /src/main_learning_train_test.py
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

