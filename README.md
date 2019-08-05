# Matching Features without Descriptors: Implicitly Matched Interest Points

![render_kitti](doc/render_kitti.gif)
![render_euroc](doc/render_euroc.gif)

This is the code for the 2019 BMVC paper **Matching Features without Descriptors: Implicitly Matched Interest Points** by Titus Cieslewski, Michael Bloesch and Davide Scaramuzza. When using this, please cite:

```bib
@InProceedings{Cieslewski19bmvc,
  author        = {Titus Cieslewski and Michael Bloesch and Davide Scaramuzza},
  title         = {Matching Features without Descriptors:
                  Implicitly Matched Interest Points},
  booktitle     = {British Machine Vision Conference (BMVC)},
  year          = 2019
}
```

## Installation

We recommend working in a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) (also when using ROS/catkin)
```bash
pip install --upgrade opencv-contrib-python==3.4.2.16 opencv-python==3.4.2.16 ipython \
    pyquaternion scipy absl-py hickle matplotlib sklearn tensorflow-gpu cachetools
```

### With ROS/catkin

```bash
sudo apt install python-catkin-tools
mkdir -p imips_ws/src
cd imips_ws
catkin config --init --mkdirs --extend /opt/ros/<YOUR VERSION> --merge-devel
cd src
git clone git@github.com:catkin/catkin_simple.git
git clone git@github.com:uzh-rpg/imips_open.git
git clone git@github.com:uzh-rpg/imips_open_deps.git
catkin build
. ../devel/setup.bash
```

### Without ROS/catkin

```bash
mkdir imips_ws
cd imips_ws
git clone git@github.com:uzh-rpg/imips_open.git
git clone git@github.com:uzh-rpg/imips_open_deps.git
```
Make sure `imips_open_deps/rpg_common_py/python`, `imips_open_deps/rpg_datasets_py/python` and `imips_open/python` are in your `PYTHONPATH`.

### Get pre-trained weights

Download the weights from http://rpg.ifi.uzh.ch/datasets/imips/tds=tm_ds=kt_d=14_ol=0.30.zip and extract them into `python/imips/checkpoints`.

## Inference

### Infer any image folder

```bash
python infer_folder.py --in_dir=INPUT_DIR [--out_dir=OUTPUT_DIR] [--ext=.EXTENSION]
```

If no output directory is provided, it will be `$HOME/imips_out/INPUT_DIR`.
`ext` can be used to specify image extensions other than `.jpg` or `.png` (add the dot).

### Test using our data

Follow [these instructions](https://github.com/uzh-rpg/imips_open_deps/tree/master/rpg_datasets_py) to link up KITTI. To speed things up, you can download http://rpg.ifi.uzh.ch/datasets/imips/tracked_indices.zip and extract the contained files to `python/imips/tracked indices` (visual overlap precalculation). Then, run:
```bash
python render_matching.py --val_best --testing
```
This will populate `results/match_render/tds=tm_ds=kt_d=14_ol=0.30_kt_testing` with images like the following:

![kt00 275 286](doc/kt00_275_286.png)

## Training

(Re)move the previously downloaded checkpoints. Follow [these instructions](https://github.com/uzh-rpg/imips_open_deps/tree/master/rpg_datasets_py) to link up TUM mono. Then, run:
```bash
python train.py
```

To visualize training progress, you can run:
```bash
python plot_val_metrics.py
```
in parallel. Here is what it should look like after around 60k iterations:

![plot_val_metrics](doc/plot_val_metrics.png)

Note that inlier counts drop initially. This is normal. With some initializations, the training seems to fail, you might need to give it some attempts. You can use the `rr` flag to change the seed.
