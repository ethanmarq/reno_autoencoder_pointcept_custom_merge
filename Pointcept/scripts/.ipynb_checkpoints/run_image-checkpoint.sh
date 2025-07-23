#!/usr/bin/env bash

apptainer exec \
    --nv \
    --bind /data/goose/:/workspace/Pointcept/data/goose/,${HOME}/git/Pointcept/:/workspace/Pointcept/ \
    pointcept.sif \
    bash