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
# Build an image and start a development container
docker compose -f .devcontainer/gpu/docker-compose.yml build --no-cache
docker compose -f .devcontainer/gpu/docker-compose.yml up -d
# Open a shell inside that container (multiple shells can be opened in one container)
docker exec -it madness bash
# Execute any main script in that shell
python workspace/src/main_mppi.py
python workspace/src/main_learning_train_test.py
# Exit from the shell/container when done
exit
# Close the container from the host machine
docker-compose -f .devcontainer/gpu/docker-compose.yml down
```

```
jupyter notebook --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --no-browser --allow-root

```

This should produce a result in the ```logs``` directory. See the ```./docs/reproducibility/``` folder for screenshots showing what the installation process should look like.

## Links

For 3D visualization:

- https://www.vpython.org/
- https://www.panda3d.org/features/
- https://github.com/fwilliams/point-cloud-utils
- https://towardsdatascience.com/python-libraries-for-mesh-and-point-cloud-visualization-part-1-daa2af36de30
- https://towardsdatascience.com/python-libraries-for-mesh-point-cloud-and-data-visualization-part-2-385f16188f0f

This one-liner loads in trajectories to Blender:

```
import bpy, numpy as np; d=np.load(r"C:\Users\moose\Desktop\dev\madness\logs\benchmark-chamber.obj_2025-02-06-15-05-57\mppi_agent\ep17\environment\state_trajectories.npz")["arr_0"][:,:3]; d[:,[0,1]] = d[:,[1,0]] * [-1,1]; c=bpy.data.curves.new("TrajectoryCurve", 'CURVE'); c.dimensions='3D'; s=c.splines.new('POLY'); s.points.add(len(d)-1); [s.points[i].co.__setitem__(slice(None), (*d[i],1.0)) for i in range(len(d))]; o=bpy.data.objects.new("Trajectory", c); bpy.context.collection.objects.link(o); bpy.context.view_layer.objects.active=o; o.select_set(True); print("Curve created!")

```

Data:

- https://github.com/subtchallenge/systems_finals_ground_truth

## Handy commands

To remove all logs (minus the gitkeep file)
```
rm -rf logs/*
```

