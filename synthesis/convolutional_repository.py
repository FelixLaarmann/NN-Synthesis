#!/usr/bin/env python

import numeric_optics.para

from itertools import product

from numeric_optics.para.convolution import *

from typing import Any
from clsp import (
    DSL,
    Constructor,
    LVar,
)

from linear_repository import Linear_Repository


class Experiment_Repository:

    def __init__(self, base_repo: Linear_Repository, kernel_shapes: list[tuple[int, int]], image_size: tuple[int, int],
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
        pairs = zip(range(1, image_size[0] + 1, 1), range(1, image_size[1] + 1, 1))
        self.convolutional_shapes: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = \
            [((w, h, i), (nw, nh, o)) for (((w, h), i), ((nw, nh), o)) in
             product(product(pairs, input_channels), product(pairs, output_channels))]

    def delta(self) -> dict[str, list[Any]]:
        return self.base_repo.delta() | {
            "convolutional_shape": self.convolutional_shapes,
            "kernel_shape": self.kernel_shapes,
            "pool_size": self.pool_sizes,
        }

    def gamma(self):
        return {
            "Layer_correlate_2d": DSL()
            .Use("cs", "convolutional_shape")
            .Use("k", "kernel_shape")
            .With(lambda cs, k: cs[0][0] - k[0] + 1 == cs[1][0] and
                                cs[0][1] - k[1] + 1 == cs[1][1])
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("activation", Constructor("Activation", LVar("af")))
            .Use("weights", Constructor("Weights", Constructor("Random") & LVar("wf")))
            .In(Constructor("Layer") & Constructor("Correlate_2D",
                                                   LVar("cs") & LVar("k") &
                                                   LVar("af") & LVar("wf"))),
            "Layer_correlate_2d_bias": DSL()
            .Use("cs", "convolutional_shape")
            .Use("k", "kernel_shape")
            .With(lambda cs, k: cs[0][0] - k[0] + 1 == cs[1][0] and
                                cs[0][1] - k[1] + 1 == cs[1][1])
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("activation", Constructor("Activation", LVar("af")))
            .Use("weights", Constructor("Weights", Constructor("Random") & LVar("wf")))
            .Use("bias", Constructor("Bias"))
            .In(Constructor("Layer") & Constructor("Bias") & Constructor("Correlate_2D",
                                                                         LVar("cs") & LVar("k") &
                                                                         LVar("af") & LVar("wf"))),
            "Layer_max_pool_2d": DSL()
            .Use("cs", "convolutional_shape")
            .Use("p", "pool_size")
            .With(lambda cs, p: cs[0][2] == cs[1][2] and
                                cs[0][0] // p[0] == cs[1][0] and
                                cs[0][1] // p[1] == cs[1][1])
            .In(Constructor("Layer") & Constructor("Max_Pool_2D",
                                                   LVar("cs") & LVar("p"))),
            "Flatten": DSL()
            .Use("cs", "convolutional_shape")
            .With(lambda cs: cs[0] == cs[1])
            .Use("s", "shape")
            .As(lambda cs: (cs[0][0] * cs[0][1] * cs[0][2], cs[1][0] * cs[1][1] * cs[1][2]))
            .In(Constructor("Layer") & Constructor("Flatten", LVar("cs") & LVar("s"))),
            "Network_Convolutional_Cons_Flatten": DSL()
            .Use("m", "layer")
            .Use("n", "layer")
            .As(lambda m: m - 1)
            .Use("cs", "convolutional_shape")
            .Use("s", "shape")
            .With(lambda cs, s: cs[1][0] * cs[1][1] * cs[1][2] == s[0])
            .Use("k", "kernel_shape")
            .Use("p", "pool_size")
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("layer", Constructor("Layer") & Constructor("Flatten", LVar("cs") & LVar("s")))
            .Use("model", Constructor("Model_Dense", LVar("n") & LVar("s") & LVar("af") & LVar("wf")))
            .In(Constructor("Model_Convolutional", LVar("m") & LVar("cs") & LVar("s") &
                            LVar("k") & LVar("p") & LVar("af") & LVar("wf"))),
            "Network_Convolutional_Cons_MaxPool": DSL()
            .Use("m", "layer")
            .Use("n", "layer")
            .As(lambda m: m - 1)
            .Use("s1", "convolutional_shape")
            .Use("s2", "convolutional_shape")
            .Use("s3", "convolutional_shape")
            .With(lambda s1, s2, s3: s3[0] == s1[0] and s1[1] == s2[0] and s3[1] == s2[1])
            .Use("k", "kernel_shape")
            .Use("p", "pool_size")
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("layer", Constructor("Layer") & Constructor("Max_Pool_2D", LVar("s1") & LVar("p")))
            .Use("model", Constructor("Model_Convolutional", LVar("n") & LVar("s2") &
                                      LVar("k") & LVar("p") & LVar("af") & LVar("wf")))
            .In(Constructor("Model_Convolutional", LVar("m") & LVar("s3") &
                            LVar("k") & LVar("p") & LVar("af") & LVar("wf"))),
            "Network_Convolutional_Cons_Correlate": DSL()
            .Use("m", "layer")
            .Use("n", "layer")
            .As(lambda m: m - 1)
            .Use("s1", "convolutional_shape")
            .Use("s2", "convolutional_shape")
            .Use("s3", "convolutional_shape")
            .With(lambda s1, s2, s3: s3[0] == s1[0] and s1[1] == s2[0] and s3[1] == s2[1])
            .Use("k", "kernel_shape")
            .Use("p", "pool_size")
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("layer", Constructor("Layer") & Constructor("Correlate_2D",
                                                             LVar("s1") & LVar("k") &
                                                             LVar("af") & LVar("wf")))
            .Use("model", Constructor("Model_Convolutional", LVar("n") & LVar("s2") &
                                      LVar("k") & LVar("p") & LVar("af") & LVar("wf")))
            .In(Constructor("Model_Convolutional", LVar("m") & LVar("s3") &
                            LVar("k") & LVar("p") & LVar("af") & LVar("wf"))),
        }

    def para_lens_algebra(self):
        return {
            "Layer_correlate_2d": (lambda cs, k, af, wf, activation, weights:
                                   ParaInit(lambda: weights((cs[1][2],) + k + (cs[0][2],)),
                                            Para(convolution.multicorrelate)) >> activation),
            "Layer_correlate_2d_bias": (lambda cs, k, af, wf, activation, weights, bias:
                                        ParaInit(lambda: weights((cs[1][2],) + k + (cs[0][2],)),
                                                 Para(convolution.multicorrelate)) >> bias(cs[1][2]) >> activation),
            "Layer_max_pool_2d": (lambda cs, p: max_pool_2d(p[0], p[1])),
            "Flatten": (lambda cs, s: flatten),
            "Network_Convolutional_Cons_Flatten": (lambda m, n, cs, s, k, p, af, wf, layer, model: layer >> model),
            "Network_Convolutional_Cons_MaxPool": (lambda m,n, s1, s2, s3, k, p, af, wf, layer, model: layer >> model),
            "Network_Convolutional_Cons_Correlate":
                (lambda m, n, s1, s2, s3, k, p, af, wf, layer, model: layer >> model),
        }
