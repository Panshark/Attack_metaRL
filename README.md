# Attack on Meta-Learning Pytorch Implementation
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Original implementation by Tristan Deleu https://github.com/tristandeleu/pytorch-maml-rl.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
	- [Contributors](#contributors)
- [License](#license)

## Background

Standard Readme started with the issue originally posed by [@maxogden](https://github.com/maxogden) over at [feross/standard](https://github.com/feross/standard) in [this issue](https://github.com/feross/standard/issues/141), about whether or not a tool to standardize readmes would be useful. A lot of that discussion ended up in [zcei's standard-readme](https://github.com/zcei/standard-readme/issues/1) repository. While working on maintaining the [IPFS](https://github.com/ipfs) repositories, I needed a way to standardize Readmes across that organization. This specification started as a result of that.

> Your documentation is complete when someone can use your module without ever
having to look at its code. This is very important. This makes it possible for
you to separate your module's documented interface from its internal
implementation (guts). This is good because it means that you are free to
change the module's internals as long as the interface remains the same.

> Remember: the documentation, not the code, defines what a module does.

~ [Ken Williams, Perl Hackers](http://mathforum.org/ken/perl_modules.html#document)

Writing READMEs is way too hard, and keeping them maintained is difficult. By offloading this process - making writing easier, making editing easier, making it clear whether or not an edit is up to spec or not - you can spend less time worrying about whether or not your initial documentation is good, and spend more time writing and using code.

By having a standard, users can spend less time searching for the information they want. They can also build tools to gather search terms from descriptions, to automatically run example code, to check licensing, and so on.

The goals for this repository are:

1. A well defined **specification**. This can be found in the [Spec document](spec.md). It is a constant work in progress; please open issues to discuss changes.
2. **An example README**. This Readme is fully standard-readme compliant, and there are more examples in the `example-readmes` folder.
3. A **linter** that can be used to look at errors in a given Readme. Please refer to the [tracking issue](https://github.com/RichardLitt/standard-readme/issues/5).
4. A **generator** that can be used to quickly scaffold out new READMEs. See [generator-standard-readme](https://github.com/RichardLitt/generator-standard-readme).
5. A **compliant badge** for users. See [the badge](#badge).

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
 - mujoco 200
 - mujoco-py 2.0.2.13

## Usage

```sh
#Train:
python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_navigation --seed 1 --num-workers 8 --use-cuda
```

```sh
#Test:
python test.py --config 2d_navigation/config.json --policy 2d_navigation/policy.th --output 2d_navigation/results.npz --num-batches 10 --meta-batch-size 20 --num-workers 12 --use-cuda
```

```sh
#Draw:
python draw.py --resultpath 2d_navigation --output_folder 2d_navigation/returns --num-batches 10 --num-traj 20 
```

```sh
#Bi-level Attack:
python train_attacker_bi_modified.py --config configs/maml/2d-navigation.yaml --output-folder 2d_navigation_bi --seed 1 --num-workers 8 --use-cuda
```

```sh
#Two-timescale Attack:
python train_one_step_attacker_modified.py --config configs/maml/2d-navigation.yaml --output-folder 2d_navigation_tt --seed 1 --num-workers 8 --use-cuda
```

## Related Efforts

- [Art of Readme](https://github.com/noffle/art-of-readme) - 💌 Learn the art of writing quality READMEs.
- [open-source-template](https://github.com/davidbgk/open-source-template/) - A README template to encourage open-source contributions.

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
