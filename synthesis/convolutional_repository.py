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
from clsp.types import Literal
from linear_repository import Linear_Repository, tails
from numeric_optics.supervised import supervised_step


class Convolutional_Repository:

    def __init__(self, base_repo: Linear_Repository,
                 convolutional_layers: [int],
                 kernel_shapes: list[tuple[int, int]],
                 image_size: tuple[int, int],
                 #input_channels: list[int],
                 #output_channels: list[int],
                 channels: list[int],
                 pool_sizes: list[tuple[int, int]]):
        self.learning_rates = base_repo.learning_rates
        self.shapes = base_repo.shapes
        self.base_repo = base_repo
        self.kernel_shapes = kernel_shapes
        #self.input_channels = input_channels
        #self.output_channels = output_channels
        self.channels = channels
        self.pool_sizes = pool_sizes
        self.convolutional_shapes: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = \
            [((w, h, i), (nw, nh, o)) for (((w, h), i), ((nw, nh), o)) in
             product(product(zip(range(1, image_size[0] + 1, 1), range(1, image_size[1] + 1, 1)), channels),
                     product(zip(range(1, image_size[0] + 1, 1), range(1, image_size[1] + 1, 1)), channels))]
        self.activation_lists = base_repo.activation_lists
        self.initialization_lists = base_repo.initialization_lists
        self.min_layers = min(convolutional_layers)
        self.max_layers = [*range(0, self.min_layers, 1)] + convolutional_layers

    def delta(self) -> dict[str, list[Any]]:
        return {
            # TODO: convolutional shapes are infeasably large and make synthesis impossible. Find clever ways to restrict this set
            "convolutional_shape": [((28,28,1), (26,26,3)), ((28,28,1), (5,5,5)), ((26,26,3), (13,13,3)), ((26,26,3), (5,5,5)),  ((13, 13, 3), (5, 5, 5)), ((13,13,3), (10,10,5)), ((10,10,5), (5,5,5)), ((5,5,5), (5,5,5))], #self.convolutional_shapes,
            "kernel_shape": self.kernel_shapes,
            "pool_size": self.pool_sizes,
            "convolutional_layer": self.max_layers
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
            .In(Constructor("Layer") & Constructor("Flatten", LVar("cs"))),

            "Network_Convolutional_Cons_Flatten": DSL()
            #.Use("m", "layer")
            .Use("n", "layer")
            #.As(lambda m: m - 1)
            .Use("cs", "convolutional_shape")
            .Use("s", "shape")
            .With(lambda cs, s: cs[1][0] * cs[1][1] * cs[1][2] == s[0])
            #.Use("k", "kernel_shape")
            #.Use("p", "pool_size")
            .Use("al", "activation_list")
            .Use("wl", "initialization_list")
            .Use("layer", Constructor("Layer") & Constructor("Flatten", LVar("cs")))
            .Use("model", Constructor("Model_Dense", LVar("n") & LVar("s") & LVar("al") & LVar("wl")))
            .In(Constructor("Model_Convolutional", Literal(0, "convolutional_layer") & LVar("cs") &
                            #LVar("k") & LVar("p") &
                            LVar("al") & LVar("wl"))
                &
                Constructor("Model_Dense", LVar("n") & LVar("s") & LVar("al") & LVar("wl"))),

            "Network_Convolutional_Cons_MaxPool": DSL()
            .Use("m", "convolutional_layer")
            .Use("n", "convolutional_layer")
            .As(lambda m: m - 1)
            .Use("n_dense", "layer")
            #.With(lambda m, n_dense: n_dense < m)
            #.Use("s1", "convolutional_shape")
            #.Use("s2", "convolutional_shape")
            #.Use("s3", "convolutional_shape")
            #TODO: compute s1 and s2 from s3 and p, because we can infer the cut type from maxpool shape computation using its pool size
            #.With(lambda s1, s2, s3: s3[0] == s1[0] and s1[1] == s2[0] and s3[1] == s2[1])
            #.Use("k", "kernel_shape")
            .Use("p", "pool_size")
            .Use("s3", "convolutional_shape")
            .Use("s1", "convolutional_shape")
            .As(lambda p, s3: (s3[0], (s3[0][0] // p[0], s3[0][1] // p[1], s3[0][2])))
            .Use("s2", "convolutional_shape")
            .As(lambda s3, s1: (s1[1], s3[1]))
            .Use("al", "activation_list")
            .Use("wl", "initialization_list")
            .Use("ad", "activation_list")
            .With(lambda al, ad: ad in tails(al))
            .Use("wd", "initialization_list")
            .With(lambda wl, wd: wd in tails(wl))
            .Use("s", "shape")
            .Use("layer", Constructor("Layer") & Constructor("Max_Pool_2D", LVar("s1") & LVar("p")))
            .Use("model", Constructor("Model_Convolutional", LVar("n") & LVar("s2") & #LVar("k") & LVar("p") &
                                      LVar("al") & LVar("wl")) &
                 Constructor("Model_Dense", LVar("n_dense") & LVar("s") & LVar("ad") & LVar("wd")))
            .In(Constructor("Model_Convolutional", LVar("m") & LVar("s3") &
                            #LVar("k") & LVar("p") &
                            LVar("al") & LVar("wl"))
                &
                Constructor("Model_Dense", LVar("n_dense") & LVar("s") & LVar("ad") & LVar("wd"))),

            "Network_Convolutional_Cons_Correlate": DSL()
            .Use("m", "convolutional_layer")
            .Use("n", "convolutional_layer")
            .As(lambda m: m - 1)
            .Use("n_dense", "layer")
            #.With(lambda m, n_dense: n_dense < m)
            #.Use("s1", "convolutional_shape")
            #.Use("s2", "convolutional_shape")
            #.Use("s3", "convolutional_shape")
            # TODO: compute s1 and s2 from s3 and k, because we can infer the cut type from correlate_2D shape computation using its kernel ----> This is not as easy, as I thought, because a correlate layer can change channels and these can't be computed from kernels alone
            #.With(lambda s1, s2, s3: s1[0] == s3[0] and s1[1] == s2[0] and s2[1] == s3[1])
            .Use("k", "kernel_shape")
            .Use("s3", "convolutional_shape")
            .Use("s1", "convolutional_shape")
            # TODO: Ask Andreas about relation between input and output channels of a convolutional layer. This +2 comes from exoeriments.convolutional
            .As(lambda k, s3: (s3[0], (s3[0][0] - k[0] + 1, s3[0][1] - k[1] + 1, s3[0][2] + 2)))
            .Use("s2", "convolutional_shape")
            .As(lambda s3, s1: (s1[1], s3[1]))
            #.Use("p", "pool_size")
            .Use("af", "activation_feature")
            .Use("atl", "activation_list")
            .Use("al", "activation_list")
            .As(lambda af, atl: (af,) + atl)
            .Use("wf", "initialization_feature")
            .Use("wtl", "initialization_list")
            .Use("wl", "initialization_list")
            .As(lambda wf, wtl: (wf,) + wtl)
            .Use("ad", "activation_list")
            .With(lambda atl, ad: ad in tails(atl))
            .Use("wd", "initialization_list")
            .With(lambda wtl, wd: wd in tails(wtl))
            .Use("s", "shape")
            .Use("layer", Constructor("Layer") & Constructor("Correlate_2D",
                                                             LVar("s1") & LVar("k") &
                                                             LVar("af") & LVar("wf")))
            .Use("model", Constructor("Model_Convolutional", LVar("n") & LVar("s2") &
                                      #LVar("k") & LVar("p") &
                                      LVar("atl") & LVar("wtl")) &
                 Constructor("Model_Dense", LVar("n_dense") & LVar("s") & LVar("ad") & LVar("wd")))
            .In(Constructor("Model_Convolutional", LVar("m") & LVar("s3") &
                            #LVar("k") & LVar("p") &
                            LVar("al") & LVar("wl"))
                &
                Constructor("Model_Dense", LVar("n_dense") & LVar("s") & LVar("ad") & LVar("wd"))),

            "Learner_Convolutional": DSL()
            .Use("n", "convolutional_layer")
            #.With(lambda n: n >= self.min_layers)
            .Use("n_dense", "layer")
            .Use("s", "shape")
            .Use("cs", "convolutional_shape")
            .Use("lr", "learning_rate")
            .Use("lrf", "learning_rate_feature")
            .Use("lf", "loss_feature")
            .Use("uf", "update_feature")
            .Use("al", "activation_list")
            .Use("wl", "initialization_list")
            .Use("ad", "activation_list")
            .With(lambda al, ad: ad in tails(al))
            .Use("wd", "initialization_list")
            .With(lambda wl, wd: wd in tails(wl))
            #.Use("k", "kernel_shape")
            #.Use("p", "pool_size")
            .Use("rate", Constructor("Learning_Rate", LVar("lrf") & LVar("lr")))
            .Use("loss", Constructor("Loss", LVar("lf")))
            .Use("upd", Constructor("Update", LVar("uf")))
            .Use("net", Constructor("Model_Convolutional", LVar("n") & LVar("cs") & LVar("al") & LVar("wl")) &
                 Constructor("Model_Dense", LVar("n_dense") & LVar("s") & LVar("ad") & LVar("wd")))
            .In(Constructor("Learner",
                            LVar("lrf") & LVar("lr") & LVar("lf") &
                            LVar("uf")) &
                Constructor("Dense", LVar("n_dense") & LVar("s") & LVar("ad") & LVar("wd")) &
                Constructor("Convolutional", LVar("n") & LVar("cs") & #LVar("k") & LVar("p") &
                            LVar("al") & LVar("wl"))),
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
            "Flatten": (lambda cs: flatten),
            "Network_Convolutional_Cons_Flatten": (lambda n, cs, s, al, wl, layer, model: layer >> model),
            "Network_Convolutional_Cons_MaxPool": (lambda m, n, n_dense, s1, s2, s3, p, al, wl, ad, wd, s, layer, model: layer >> model),
            "Network_Convolutional_Cons_Correlate":
                (lambda m, n, n_dense, s1, s2, s3, k, af, atl, al, wf, wtl, wl, ad, wd, s, layer, model: layer >> model),
            "Learner_Convolutional": (lambda n, m, s, cs, lr, lrf, lf, uf, al, wl, ad, wd, rate, loss, upd, net:
                                      (supervised_step(net, upd, loss, rate), net))
        }
