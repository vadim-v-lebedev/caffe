#!/usr/bin/env sh

../../build/tools/caffe train --solver=solver_tune2.prototxt --weights=$1
