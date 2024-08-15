#!/usr/bin/env python

import numeric_optics.para

from itertools import product

import numeric_optics.lens as lens
from numeric_optics.para import Para, to_para, linear, to_para_init
from numeric_optics.supervised import supervised_step, mse_loss, learning_rate, rda_learning_rate
from numeric_optics.update import gd, rda, rda_momentum, momentum
from numeric_optics.initialize import normal, glorot_normal, glorot_uniform

from typing import Any
from clsp import (
    DSL,
    Constructor,
    LVar,
)
from clsp.types import Literal


class Linear_Repository:

    def __init__(self, learning_rates: list[float],
                 input_neurons: int, output_neurons: int,
                 hidden_layers: [int], hidden_neurons: [int],
                 train_input, train_labels):
        self.learning_rates = learning_rates
        self.min_layers = min(hidden_layers)
        self.max_layers = [*range(0, self.min_layers, 1)] + hidden_layers  # [*range(1, max_layers + 1, 1)]
        self.shapes = list(product([input_neurons, output_neurons] + hidden_neurons,
                                   [input_neurons, output_neurons] + hidden_neurons))
        self.train_input = train_input
        self.train_labels = train_labels

    def delta(self) -> dict[str, list[Any]]:
        return {
            "learning_rate": self.learning_rates,
            "learning_rate_feature": ["Constant", "RDA"],
            "loss_feature": ["MSE"],
            "update_feature": ["Momentum", "Gradient_Descent", "RDA"],
            "activation_feature": ["Sigmoid", "ReLu"],
            "initialization_feature": ["Glotrot_Uniform", "Glotrot_Normal", "Normal"],
            "layer": self.max_layers,
            "shape": self.shapes
        }

    def gamma(self):
        return {
            "Learning_Rate": DSL()
            .Use("lr", "learning_rate")
            .In(Constructor("Learning_Rate", Literal("Constant", "learning_rate_feature") & LVar("lr"))),
            "Learning_Rate_RDA": DSL()
            .Use("lr", "learning_rate")
            .In(Constructor("Learning_Rate", Literal("RDA", "learning_rate_feature") & LVar("lr"))),
            "Loss_MSE": Constructor("Loss", Literal("MSE", "loss_feature")),
            "Update_RDA": Constructor("Update", Literal("RDA", "update_feature")),
            "Update_RDA_Momentum": Constructor("Update",
                                               Literal("RDA", "update_feature") &
                                               Literal("Momentum", "update_feature")),
            "Update_GradientDescent": Constructor("Update", Literal("Gradient_Descent", "update_feature")),
            "Update_GradientDescent_Momentum": Constructor("Update",
                                                           Literal("Gradient_Descent", "update_feature") &
                                                           Literal("Momentum", "update_feature")),
            "Activation_Sigmoid": Constructor("Activation", Literal("Sigmoid", "activation_feature")),
            "Activation_ReLu": Constructor("Activation", Literal("ReLu", "activation_feature")),
            "Weights_Initial_Normal": Constructor("Weights",
                                                  Constructor("Random") & Literal("Normal", "initialization_feature")),
            "Weights_Initial_GlotrotUniform": Constructor("Weights",
                                                          Constructor("Random") &
                                                          Literal("Glotrot_Uniform", "initialization_feature")),
            "Weights_Initial_Glotrot": Constructor("Weights",
                                                   Constructor("Random") &
                                                   Literal("Glotrot_Normal", "initialization_feature")),
            "Bias": Constructor("Bias"),
            "Layer_Dense": DSL()
            .Use("s", "shape")
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("activation", Constructor("Activation", LVar("af")))
            .Use("weights", Constructor("Weights", Constructor("Random") & LVar("wf")))
            .Use("bias", Constructor("Bias"))
            .In(Constructor("Layer") & Constructor("Dense", LVar("s") & LVar("af") & LVar("wf"))),
            "Network_Dense_Start": DSL()
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("s", "shape")
            .Use("_input", Constructor("Layer") & Constructor("Dense", LVar("s") & LVar("af") & LVar("wf")))
            .In(Constructor("Model", Literal(0, "layer") & LVar("s") & LVar("af") & LVar("wf"))),
            "Network_Dense": DSL()
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("m", "layer")
            .Use("n", "layer")
            .As(lambda m: m - 1)
            .Use("s1", "shape")
            .Use("s2", "shape")
            .Use("s3", "shape")
            .With(lambda s1, s2, s3: s3[0] == s1[0] and s1[1] == s2[0] and s3[1] == s2[1])
            .Use("layer", Constructor("Layer") & Constructor("Dense", LVar("s1") & LVar("af") & LVar("wf")))
            .Use("model", Constructor("Model", LVar("n") & LVar("s2") & LVar("af") & LVar("wf")))
            .In(Constructor("Model", LVar("m") & LVar("s3") & LVar("af") & LVar("wf"))),
            "Learner": DSL()
            .Use("n", "layer")
            .With(lambda n: n >= self.min_layers)
            .Use("s", "shape")
            .Use("lr", "learning_rate")
            .Use("lrf", "learning_rate_feature")
            .Use("lf", "loss_feature")
            .Use("uf", "update_feature")
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("rate", Constructor("Learning_Rate", LVar("lrf") & LVar("lr")))
            .Use("loss", Constructor("Loss", LVar("lf")))
            .Use("upd", Constructor("Update", LVar("uf")))
            .Use("net", Constructor("Model", LVar("n") & LVar("s") & LVar("af") & LVar("wf")))
            .In(Constructor("Learner",
                            LVar("lrf") & LVar("lr") & LVar("lf") & LVar("uf") &
                            LVar("n") & LVar("s") & LVar("af") & LVar("wf"))),
        }

    @staticmethod
    def layer_dense(shape, af, wf, activation, weights, bias):
        return linear(shape, weights) >> bias(shape[1]) >> activation

    def para_lens_algebra(self):
        return {
            "Learning_Rate": (lambda n: to_para(learning_rate(n))),
            "Learning_Rate_RDA": (lambda n: to_para(rda_learning_rate(n))),
            "Loss_MSE": Para(mse_loss),
            "Update_RDA": rda,
            "Update_RDA_Momentum": rda_momentum(-0.1),
            "Update_GradientDescent": gd(-0.01),
            "Update_GradientDescent_Momentum": momentum(-0.01, -0.1),
            "Activation_Sigmoid": to_para_init(lens.sigmoid),
            "Activation_ReLu": to_para_init(lens.relu),
            "Weights_Initial_Normal": normal(0, 0.01),
            "Weights_Initial_GlotrotUniform": glorot_uniform,
            "Weights_Initial_Glotrot": glorot_normal,
            "Bias": numeric_optics.para.bias,
            "Layer_Dense": self.layer_dense,
            "Network_Dense_Start": (lambda af, wf, s, l: l),
            "Network_Dense": (lambda af, wf, m, n, s1, s2, s3, layer, model: layer >> model),
            "Learner": (lambda n, s, lr, lrf, lf, uf, af, wf, rate, loss, upd, net:
                        (supervised_step(net, upd, loss, rate), net)),
        }
