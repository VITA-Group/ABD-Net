# Data Preparation and Miscs

Refer to original README [here](./README_ORIG.md).

# Cython Evalution

```bash
cd torchreid/eval_cylib
make
```

# Command

Example:

```bash
python train.py -s market1501 -t market1501 \
    --flip-eval --eval-freq 1 \
    --label-smooth \
    --criterion htri \
    --lambda-htri 0.1  \
    --data-augment crop random-erase \
    --margin 1.2 \
    --train-batch-size 64 \
    --height 384 \
    --width 128 \
    --optim adam --lr 0.0003 \
    --stepsize 20 40 \
    --gpu-devices 4,5 \
    --max-epoch 80 \
    --save-dir path/to/dir \
    --arch resnet50 \
    --use-of \
    --abd-dan cam pam \
    --abd-np 2 \
    --shallow-cam \
    --use-ow
```

**For testing, add extra arguments `--evaluate --load-weights path/to/checkpoint.pth.tar`**.

## Criterion

 + `--criterion`. May be `xent`, `htri`.

## OF
 + `--use-of`
 + `--of-beta <beta>`. Default `1e-6`.
 + `--of-start-epoch <epoch>`. Default `23`.
 + `--of-position <p1> <p2> ...`. Can be a subset of `{before, after, cam, pam, intermediate}`. Default to all of them.

## OW

 + `--use-ow`
 + `--ow-beta <beta>`. Default `1e-3`.

## Shallow CAM

 + `--shallow-cam`. When set, ShallowCAM will be used.

## Branches

 + `--branches <b1> <b2> ...`. Can be a subset of `{global, abd, dan, np}`. Default to `{global, abd}`.

## Global Branch

 + `--global-dim <dim>`. Default to `1024`.

## ABD Branch

 + `--abd-dim <dim>`. Specify the feature dim for each part. Default to `1024`.
 + `--abd-np <np>`. Default to `2`.
 + `--abd-dan ...`. Can be a subset of `{cam, pam}`. Default to `{}`.
 + `--abd-dan-no-head`. When set, DANHead will not be used.

## NP Branch

+ `--np-dim <dim>`. Specify the feature dim for each part. Default to `1024`.
+ `--np-np <np>`. Default to `2`.

## DAN Branch

 + `--dan-dim <dim>`. Specify the feature dim for each part. Default to `1024`.
 + `--dan-dan ...`. Can be a subset of `{cam, pam}`. Default to `{}`.
 + `--dan-dan-no-head`. When set, DANHead will not be used.

## Arch

 + `--arch <arch>`. May be one of `{resnet50, densenet121, densenet121_d4, densenet121_t3_d4, densenet121_d3_t3_d4, densenet161, densenet161_d4, densenet161_t3_d4, densenet161_d3_t3_d4}`.
