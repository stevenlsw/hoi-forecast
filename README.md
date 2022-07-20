# HOI-Forecast

**Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos (CVPR 2022)**
<a href="https://arxiv.org/abs/2204.01696"><img src="https://img.shields.io/badge/arXiv-2204.01696-b31b1b.svg"></a>

#### [[Project Page]](https://stevenlsw.github.io/hoi-forecast/) [[Paper]](https://arxiv.org/abs/2204.01696) [[Training Data]](https://drive.google.com/drive/folders/1llDYFwn2gGQLpcWy6YScp3ej7A3LIPFc)

Given observation frames of the past, we predict future hand trajectories (green and red lines) and object interaction hotspots (heatmaps) in egocentric view. We genearte training data **automatically** and use this data to train an Object-Centric Transformer (OCT) model for prediction.
<br>
<p align="center">
    <img src="assets/teaser.gif" width="456">
</p>


## Installation
- Clone this repository: 
    ```Shell
    git clone https://github.com/stevenlsw/hoi-forecast
    cd hoi-forecast
    ```
- Python 3.6 Environment:
    ```Shell
    conda env create -f environment.yaml
    conda activate fhoi
    ```

## Quick training data generation
Official Epic-Kitchens Dataset looks the same as `assets/EPIC-KITCHENS`, rgb frames needed for the demo has been pre-downloaded in `assets/EPIC-KITCHENS/P01/rgb_frames/P01_01`. 

- Download Epic-Kitchens 55 Dataset [annotations](https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-55-annotations/master/EPIC_train_action_labels.csv) and save in `assets` folder

- Download hand-object detections below
    ```Shell
    link=https://data.bris.ac.uk/datasets/3l8eci2oqgst92n14w2yqi5ytu/hand-objects/P01/P01_01.pkl
    wget -P assets/EPIC-KITCHENS/P01/hand-objects $link
    ```
- Run `python demo_gen.py` and results [png, pkl] are stored in `figs`, you should visualize the result 

<div align="center">
<img src="assets/demo_gen.jpg" width="60%">
</div>

- For more generated training labels, please visit [google drive](https://drive.google.com/drive/folders/1lNOSXXKbiqYJqC1hp1CIaIeM8u6UVvcg) and run `python example.py`. 


## Evaluation on EK100
We maunally collect the hand trajectories and interaction hotspots for evaluation. We pre-extract the input videos features.

- Download the processed [files](https://drive.google.com/file/d/1IUreVIjAKbi-TZq7ogJz1MfTvVnpV2J5/view?usp=sharing) (include collected labels, pre-extracted features, and dataset partitions, 600 MB) and **unzipped**. You will get the stucture like:
    ```
    hoi-forecast
    |-- data 
    |   |-- ek100
    |   |   |-- ek100_eval_labels.pkl
    |   |   |-- video_info.json
    |   |   |-- labels
    |   |   |   |-- label_303.pkl
    |   |   |   |-- ...
    |   |   |-- feats
    |   |   |   |-- data.lmdb (RGB)
    |-- common
    |   |-- epic-kitchens-55-annotations
    |   |-- epic-kitchens-100-annotations
    |   |-- rulstm
    ```

- Download [pretrained models](https://drive.google.com/file/d/1tqoD8fy6ty1nxclEi-YTv3ivUwcLERQA/view?usp=sharing) on EK100 and the stored model path is refered as `$resume`. 

- Install PyTorch and dependencies by the following command:
    ```Shell
    pip install -r requirements.txt
    ```

- Evaluate future hand trajectory
    ```Shell
    python traineval.py --evaluate --ek_version=ek100 --resume={path to the model} --traj_only
    ```

- Evaluate future interaction hotspots
    ```Shell
    python traineval.py --evaluate --ek_version=ek100 --resume={path to the model}
    ```

- Results should like:

    <table>
    <thead>
    <tr>
        <th colspan="2" style="text-align:center;">Hand Trajectory</th>
        <th colspan="3" style="text-align:center;">Interaction Hotspots</th>
    </tr>
    <tr>
        <th>ADE &#8595;</th>
        <th>FDE &#8595;</th>
        <th>SIM &#8593;</th>
        <th>AUC-J &#8593;</th>
        <th>NSS &#8593;</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <th>0.12</th>
        <th>0.11</th>
        <th>0.19</th>
        <th>0.69</th>
        <th>0.72</th>
    </tr>
    </tbody>
    </table>


## Training
- Extract per-frame features of training set similar to [RULSTM](https://github.com/fpv-iplab/rulstm) and store them in `data/ek100/feats/ek100.lmdb`, the key-value pair likes
    ```python
    fname = 'P01/rgb_frames/P01_01/frame_0000000720.jpg'
    env[fname.encode()] = result_dict # extracted feature results
    ``` 

- Start training
    ```
    python traineval.py --ek_version=ek100
    ``` 

## Citation
```latex
@inproceedings{liu2022joint,
  title={Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos},
  author={Liu, Shaowei and Tripathi, Subarna and Majumdar, Somdeb and Wang, Xiaolong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## Acknowledges
We thank:
* [epic-kitchens](https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes) for hand-object detections on Epic-Kitchens dataset
* [rulstm](https://github.com/fpv-iplab/rulstm) for features extraction and action anticipation
* [epic-kitchens-dataset-pytorch](https://github.com/guglielmocamporese/epic-kitchens-dataset-pytorch) for 
epic-kitchens dataloader
* [Miao Liu](https://aptx4869lm.github.io/) for help with prior work

