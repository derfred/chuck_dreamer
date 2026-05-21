# Chuck Dreamer

## Installation

Requires Python 3.12+. Uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
uv sync
```

## Usage

```bash
python main.py [--config PATH] [--verbose] COMMAND [OPTIONS]
```

## How to get started

The basic approach of this project is to train a world model based on simulation data to learn the basic pushing dynamics and then augment it with real-world data. For this purpose the project implements a customizable simulation environment.

Get a feel for the environment and the data collection process by running:
```
mjpython main.py show-scene   # on macos
python main.py show-scene     # on linux
```

This will show you a window with a simulated scene. By default it uses a very simple heuristic policy that moves the end-effector after pressing "space". There are many customization options that you can find in `src/chuck_dreamer/sim/scene_generator.py`.

The next step is to generate a dataset of simulated trajectories. This will use the heuristic policy to generate many trajectories and save them to disk. Run:

```
python main.py generate-scenes --episodes 100 --output data/scenes
```

Many of these trajectories will not be interesting, because the default heuristic policy is very simple. To select interesting trajectories both in positive and negative sense, you can use the `notebooks/sample_episodes.ipynb` notebook.

At the end of the notebook you are given a list of train and eval episodes. You should move them to the `data/train` and `data/eval` folders, respectively.

### Adding real-world data

Adding real-world data requires several data augmentation steps to add data that the simulation provides, in particular:
- Object position in the image (object_uv)
- Object position in the world (object_xy)
- Segmentation masks (segmentation_*)

The details of the data-format can be found in the EpisodeWriter class in `src/chuck_dreamer/sim/episode_writer.py`. We provide about 500 episodes of real-world data on [HuggingFace](https://huggingface.co/chorfiyoussef). In particular the following datasets are relevant (the numbers after the # in the project specific selector syntax, and they skip the calibration episodes at the beginning of the dataset):
```
chorfiyoussef/task1_2005_3_20260520_195038#11-9999
chorfiyoussef/task1_2005_2_20260520_191951#11-9999
chorfiyoussef/task1_2005_1_20260520_181642#11-9999
chorfiyoussef/task_2_1805_3#8-9999
chorfiyoussef/task_2_1805_2#8-9999
chorfiyoussef/task_2_1805_1#8-9999
chorfiyoussef/task_1_1805_6#8-9999
chorfiyoussef/task_1_1805_5#8-9999
chorfiyoussef/task_1_1805_4#8-9999
chorfiyoussef/task_1_1805_3#8-9999
chorfiyoussef/task_1_1805_2#8-9999
```

For these datasets the calibration data has already been extracted and saved in the `calibration_cache` folder in this repository. If you want to recreate them yourself, delete the dataset specific folder in `calibration_cache` and run:
```
python main.py import-lerobot <DATASET_SELECTOR> --doctor
```

Without going into excessive detail, the augmentation of the real-world data relies on 3D reconstruction of the object from the scenes and extracting 3d pose and segmentation masks. The above command will provide you with further commands for the individual steps, many of which require manual data annotation.

When a dataset is ready for training, you can use the `import-lerobot` command to convert it to the same format as the simulated data. For example:
```
python main.py import-lerobot chorfiyoussef/task_2_1805_1#8-9999 --output data/real --tag real
```

The use of the `--tag real` option add a "real" tag to the resulting episodes, they significance of which will become clear in the next section.

### Modelling & Training

This codebase implements a fairly standard Dreamer v1 style world model [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603). There are options to augment the visual reconstruction loss with additional data. In our experiments we used:
* Image reconstruction loss (image)
* Focus Masked based loss boosting based on the segmentation masks of the object
* Auxiliary loss on the predicted object position in the image (object_xy)

Affordances exist to use multi-modal inputs, but they were not systematically investigated in this project. The additional data modalities that are available are:
* Object position in the image (object_uv)
* Object position in the world (object_xy)
* End-effector position in world coordinates (ee_xy)
* Joint positions (joints)

These can be combined using the `-o env.obs_mode='["image", "object_uv", ...]` option on the train command. Which we will now explore in more detail:

```
python main.py train -o ..overrides... -c ..configfile...
```

All training options are documented in the default config file `configs/default.yaml`. For the next section we will focus on the data-mixing options, which are relevant for the purposes of augmenting the simulated data with real-world data. The relevant incantations are:

```
python main.py train -o 'data.training.protected_tags=["real"]' -o data.training.tag_weights.real=2.0
```

This will instruct the replay buffer to sample real-world episodes with twice the probability of simulated episodes. The `protected_tags` option ensures that all episodes with the "real" tag are not evicted from the replay buffer. The significance of this is that during training new simulated episodes are generated and added to the replay buffer, as we can cheaply generate these on the fly. For real-world episodes this is not the case.

Finally the trainer provides some options for scaling the training process. For this purpose a `config/h200.yaml` file is provided that has optimized settings for training on an NVIDIA H200 GPU. We provide our highest performing checkpoint here: (final.safetensors)[final.safetensors].

### Evaluation

The codebase provides Trackio and Weights & Biases integration for tracking an ongoing training run. For more detailed evaluation of a checkpoint we provide the `eval` command, which can be used as follows:

```
python main.py eval --all --checkpoint PATH_TO_CHECKPOINT -o hardware.device=mlx # override the device to eval a CUDA checkpoint on a mac for example
```

This command will execute the template notebooks in `src/chuck_dreamer/evals` on the provided checkpoint. It will also provide rerun compatible versions to inspect the reconstructions of the world model on the eval episodes. The evaluations focus on the following aspects:
* Reconstruction quality of the prior and posterior world model
* Open-loop rollout quality of the prior world model compared to the zero-dynamics baseline
* Quality of the learned reward model
* Informativeness of the learned latent space with respect to the dynamic variables of the system (object position, end-effector position, etc.) using linear probes.

### Inference

The inference topic is last because it is the least developed aspect of this project. The first step is to calibrate the camera and robot for the particular setup. Using the `python main.py annotate-live --output my_local_calibration.yaml` command you will be guided through process of calibrating the extrinsics of the camera and position of the robot arm relative to the world frame.

Given this calibration, you can then come to the main event:
```
python main.py run --checkpoint PATH_TO_CHECKPOINT --port /dev/tty.YOUR_SO101_PORT --policy cem # or dreamer if you extend the trainer
```

The `run` command will show a window with the live camera feed and provide basic control over the robot.
