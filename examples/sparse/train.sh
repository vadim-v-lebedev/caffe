#!/usr/bin/env sh

../../build/tools/caffe train --solver=solver.prototxt
cp snapshots/net_iter_2000.caffemodel net.caffemodel
