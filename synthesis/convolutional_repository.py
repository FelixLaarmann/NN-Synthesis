#!/usr/bin/env python

import numeric_optics.para

from itertools import product

import numeric_optics.lens as lens
from numeric_optics.para import Para, to_para, dense, linear, to_para_init
from numeric_optics.supervised import train_supervised, supervised_step, mse_loss, learning_rate, rda_learning_rate
from numeric_optics.update import gd, rda, rda_momentum, momentum
from numeric_optics.statistics import accuracy
from numeric_optics.initialize import normal, glorot_normal, glorot_uniform
from numeric_optics.para.convolution import *

from typing import Any
from clsp import (
    DSL,
    Constructor,
    LVar,
)
from clsp.types import Literal
from clsp.enumeration import interpret_term

from linear_repository import Linear_Repository


class Experiment_Repository:

    def __init__(self, base_repo: Linear_Repository, kernel_shapes: list[tuple[int, int]],
                 input_channels: list[int], output_channels: list[int], pool_sizes: list[tuple[int, int]]):
        self.learning_rates = base_repo.learning_rates
        self.min_layers = base_repo.min_layers
        self.max_layers = base_repo.max_layers
        self.shapes = base_repo.shapes
        self.train_input = base_repo.train_input
        self.train_labels = base_repo.train_labels
        self.base_repo = base_repo
        self.kernel_shapes = kernel_shapes
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.pool_sizes = pool_sizes

    def delta(self) -> dict[str, list[Any]]:
        return self.base_repo.delta() | {
            "kernel_shape": self.kernel_shapes,
            "input_channel": self.input_channels,
            "output_channel": self.output_channels,
            "pool_size": self.pool_sizes,
        }

    def gamma(self):
        return {
            "Layer_correlate_2d": DSL()
            .Use("k", "kernel_shape")
            .Use("in", "input_channel")
            .Use("out", "output_channel")
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("activation", Constructor("Activation", LVar("af")))
            .Use("weights", Constructor("Weights", Constructor("Random") & LVar("wf")))
            .In(Constructor("Layer") & Constructor("Correlate_2D",
                                                   LVar("k") & LVar("in") & LVar("out") &
                                                   LVar("af") & LVar("wf"))),
            "Layer_correlate_2d_biased": DSL()
            .Use("k", "kernel_shape")
            .Use("in", "input_channel")
            .Use("out", "output_channel")
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("activation", Constructor("Activation", LVar("af")))
            .Use("weights", Constructor("Weights", Constructor("Random") & LVar("wf")))
            .Use("bias", Constructor("Bias"))
            .In(Constructor("Layer") & Constructor("Correlate_2D",
                                                   LVar("k") & LVar("in") & LVar("out") &
                                                   LVar("af") & LVar("wf"))),
            "Layer_max_pool_2d": DSL()
            .Use("p", "pool_size")
            .In(Constructor("Layer") & Constructor("Max_Pool_2D",
                                                   LVar("p"))),

        }

    def para_lens_algebra(self):
        return {
            "Layer_correlate_2d": (lambda k, i, o, af, wf, activation, weights:
                                   ParaInit(lambda: weights((o,) + k + (i,)), Para(convolution.multicorrelate)) >>
                                   activation),
            "Layer_correlate_2d_biased": (lambda k, i, o, af, wf, activation, weights, bias:
                                          ParaInit(lambda: weights((o,) + k + (i,)),
                                                   Para(convolution.multicorrelate)) >>
                                          bias(o) >> activation)
        }
