# Attack on Meta-Learning Pytorch Implementation
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Original implementation by [Tristan Deleu](https://github.com/tristandeleu/pytorch-maml-rl) based on [Chelsea Finn et al.](https://arxiv.org/abs/1703.03400) in Pytorch.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [References](#references)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
	- [Contributors](#contributors)
- [License](#license)

## Background

### 2d-navigation
2d-navigation without attack             |  2d-navigation with attack
:-------------------------:|:-------------------------:
<img width="420" height="315" src="https://github.com/Panshark/Attack_metaRL/blob/main/image_source/withoutattack_2d.gif"/>  |  <img width="420" height="315" src="https://github.com/Panshark/Attack_metaRL/blob/main/image_source/attack_2d.gif"/>

### Halfcheetah-vel
Halfcheetah-vel without attack             |  Halfcheetah-vel with attack
:-------------------------:|:-------------------------:
<img width="420" height="420" src="https://github.com/Panshark/Attack_metaRL/blob/main/image_source/withoutattack_cheetah.gif"/>  |  <img width="420" height="420" src="https://github.com/Panshark/Attack_metaRL/blob/main/image_source/attack_cheetah.gif"/>

Meta learning, as a learning how to learn method, has become the focus of researchers in recent years because of its great prospect in the field of artificial intelligence. During the meta training, the meta learner can develop a common learning strategy, such as the learning experience memory modules or a fast adaptation initial value, for not only the computer vision tasks but also the reinforcement learning tasks. Despite its approvingly efficient performance, meta learning is under the suspicion of security. The dependability and robustness of meta learning are doubtful, especially in the reinforcement learning area when given an extremely complex task environment. Even though meta reinforcement learning has not been extensively applied, it would be too late to consider the potential threat when attacks occur. Therefore, in our paper, we create an adversarial attacking method on the sampling process of meta reinforcement learning.

## Install

To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a conda environment with [`Anaconda`](https://www.anaconda.com/). To create a conda environment:
```
conda create -n your_env_name python=3.8
```
Activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
conda activate your_env_name
pip install -r requirements.txt
```

#### Requirements
 - Python 3.8 or above
 - PyTorch 1.9.1
 - MuJoCo 200
 - mujoco-py 2.0.2.13

## Usage

### Learner Training

```sh
python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_navigation --seed 1 --num-workers 8 --use-cuda
```

### Learner Testing

```sh
python test.py --config 2d_navigation/config.json --policy 2d_navigation/policy.th --output 2d_navigation/results.npz --num-batches 10 --meta-batch-size 20 --num-workers 12 --use-cuda
```

### Learner Result

```sh
python draw.py --resultpath 2d_navigation --output_folder 2d_navigation/returns --num-batches 10 --num-traj 20 
```

### Intermittent Attacker Training

```sh
python train_intermittent.py --config configs/maml/2d-navigation.yaml --output-folder 2d_navigation_inter --seed 1 --num-workers 8 --use-cuda
```

### Persistent Attacker Training

```sh
python train_persistent.py --config configs/maml/2d-navigation.yaml --output-folder 2d_navigation_pers --seed 1 --num-workers 8 --use-cuda
```

## References
This project contains experiments implementation of
> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep
Networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]

If you want to cite the paper
```
@article{finn17maml,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {{Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks}},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```

## Maintainers

[@Haozhe Lei](https://github.com/Panshark).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/Panshark/Attack_metaRL/issues/new) or submit PRs.

Standard Readme follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

### Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/Panshark"><img src="https://avatars.githubusercontent.com/u/71244619?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Haozhe Lei</b></sub></a><br /><a href="https://github.com/Panshark/Attack_metaRL/commits?author=Panshark" title="Code">💻</a> <a href="#data-Panshark" title="Data">🔣</a> <a href="https://github.com/Panshark/Attack_metaRL/commits?author=Panshark" title="Documentation">📖</a> <a href="#ideas-Panshark" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-Panshark" title="Maintenance">🚧</a> <a href="#projectManagement-Panshark" title="Project Management">📆</a> <a href="#question-Panshark" title="Answering Questions">💬</a> <a href="https://github.com/Panshark/Attack_metaRL/pulls?q=is%3Apr+reviewed-by%3APanshark" title="Reviewed Pull Requests">👀</a> <a href="#design-Panshark" title="Design">🎨</a></td>
    <td align="center"><a href="https://engineering.nyu.edu/student/tao-li-0"><img src="https://avatars.githubusercontent.com/u/46550706?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Tao Li</b></sub></a><br /><a href="#design-TaoLi-NYU" title="Design">🎨</a> <a href="#eventOrganizing-TaoLi-NYU" title="Event Organizing">📋</a> <a href="#ideas-TaoLi-NYU" title="Ideas, Planning, & Feedback">🤔</a> <a href="#data-TaoLi-NYU" title="Data">🔣</a> <a href="#content-TaoLi-NYU" title="Content">🖋</a> <a href="#question-TaoLi-NYU" title="Answering Questions">💬</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## License

[MIT](LICENSE) © Haozhe Lei
