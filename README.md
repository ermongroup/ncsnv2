# Improved Techniques for Training Score-Based Generative Models

This repo contains the official implementation for the paper [Improved Techniques for Training Score-Based Generative Models](http://arxiv.org/abs/2006.09011). 

by [Yang Song](http://yang-song.github.io/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/), Stanford AI Lab.

**Note**: The method has been extended by the subsequent work [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) ([code](https://github.com/yang-song/score_sde)) that allows better sample quality and exact log-likelihood computation.

-----------------------------------------------------------------------------------------

We significantly improve the method proposed in [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600). Score-based generative models are flexible neural networks trained to capture the score function of an underlying data distribution—a vector field pointing to directions where the data density increases most rapidly. We present new techniques to improve the performance of score-based generative models, scaling them to *high resolution images* that are previously impossible. *Without requiring adversarial training*, they can produce sharp and diverse image samples that rival GANs.

![samples](assets/samples.jpg)

(**From left to right**: Our samples on FFHQ 256px, LSUN bedroom 128px, LSUN tower 128px, LSUN church_outdoor 96px, and CelebA 64px.)

## Running Experiments

### Dependencies

Run the following to install all necessary python packages for our code.

```bash
pip install -r requirements.txt
```

### Project structure

`main.py` is the file that you should run for both training and sampling. Execute ```python main.py --help``` to get its usage description:

```
usage: main.py [-h] --config CONFIG [--seed SEED] [--exp EXP] --doc DOC
               [--comment COMMENT] [--verbose VERBOSE] [--test] [--sample]
               [--fast_fid] [--resume_training] [-i IMAGE_FOLDER] [--ni]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --exp EXP             Path for saving running related data.
  --doc DOC             A string for documentation purpose. Will be the name
                        of the log folder.
  --comment COMMENT     A string for experiment comment
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  --test                Whether to test the model
  --sample              Whether to produce samples from the model
  --fast_fid            Whether to do fast fid test
  --resume_training     Whether to resume training
  -i IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The folder name of samples
  --ni                  No interaction. Suitable for Slurm Job launcher
```

Configuration files are in `config/`. You don't need to include the prefix `config/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
├── logs # contains checkpoints and samples produced during training
│   └── <doc> # a folder named by the argument `--doc` specified to main.py
│      ├── checkpoint_x.pth # the checkpoint file saved at the x-th training iteration
│      ├── config.yml # the configuration file for training this model
│      ├── stdout.txt # all outputs to the console during training
│      └── samples # all samples produced during training
├── fid_samples # contains all samples generated for fast fid computation
│   └── <i> # a folder named by the argument `-i` specified to main.py
│      └── ckpt_x # a folder of image samples generated from checkpoint_x.pth
├── image_samples # contains generated samples
│   └── <i>
│       └── image_grid_x.png # samples generated from checkpoint_x.pth       
└── tensorboard # tensorboard files for monitoring training
    └── <doc> # this is the log_dir of tensorboard
```

### Training

For example, we can train an NCSNv2 on LSUN bedroom by running the following

```bash
python main.py --config bedroom.yml --doc bedroom
```

Log files will be saved in `<exp>/logs/bedroom`.

### Sampling

If we want to sample from NCSNv2 on LSUN bedroom, we can edit `bedroom.yml` to specify the `ckpt_id` under the group `sampling`, and then run the following

```bash
python main.py --sample --config bedroom.yml -i bedroom
```

Samples will be saved in `<exp>/image_samples/bedroom`.

We can interpolate between different samples (see more details in the paper). Just set `interpolation` to `true` and an appropriate `n_interpolations` under the group of `sampling` in `bedroom.yml`. We can also perform other tasks such as inpainting. Usages should be quite obvious if you read the code and configuration files carefully.

### Computing FID values quickly for a range of checkpoints

We can specify `begin_ckpt` and `end_ckpt` under the `fast_fid` group in the configuration file. For example, by running the following command, we can generate a small number of samples per checkpoint within the range `begin_ckpt`-`end_ckpt` for a quick (and rough) FID evaluation.

```shell
python main.py --fast_fid --config bedroom.yml -i bedroom
```

You can find samples in `<exp>/fid_samples/bedroom`.

## Pretrained Checkpoints

Link: https://drive.google.com/drive/folders/1217uhIvLg9ZrYNKOR3XTRFSurt4miQrd?usp=sharing

You can produce samples using it on all datasets we tested in the paper. It assumes the `--exp` argument is set to `exp`.

## References

If you find the code/idea useful for your research, please consider citing

```bib
@inproceedings{song2020improved,
  author    = {Yang Song and Stefano Ermon},
  editor    = {Hugo Larochelle and
               Marc'Aurelio Ranzato and
               Raia Hadsell and
               Maria{-}Florina Balcan and
               Hsuan{-}Tien Lin},
  title     = {Improved Techniques for Training Score-Based Generative Models},
  booktitle = {Advances in Neural Information Processing Systems 33: Annual Conference
               on Neural Information Processing Systems 2020, NeurIPS 2020, December
               6-12, 2020, virtual},
  year      = {2020}
}
```

and/or our previous work

```bib
@inproceedings{song2019generative,
  title={Generative Modeling by Estimating Gradients of the Data Distribution},
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11895--11907},
  year={2019}
}
```

