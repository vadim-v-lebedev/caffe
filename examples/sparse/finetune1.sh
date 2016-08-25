#!/usr/bin/env sh

../../build/tools/caffe train --solver=solver_tune1.prototxt --weights=$1
