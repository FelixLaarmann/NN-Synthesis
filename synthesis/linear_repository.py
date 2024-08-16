#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import argparse

import torch 
import torch.nn as nn 

import numeric_optics.para

from itertools import product, groupby

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


def tails(ls: list[Any]) -> list[list[Any]]:
    tl = ls
    tls = []
    while tl:
        tls.append(tl)
        tl = tl[1:]
    return tls


class Linear_Repository:

    def __init__(self, learning_rates: list[float],
                 input_neurons: int, output_neurons: int,
                 hidden_layers: [int], hidden_neurons: [int],
                 activation_lists: list[list[str]],
                 initialization_lists: list[list[str]]):
        self.learning_rates = learning_rates
        self.min_layers = min(hidden_layers)
        self.max_layers = [*range(0, self.min_layers, 1)] + hidden_layers  # [*range(1, max_layers + 1, 1)]
        self.shapes = list(product([input_neurons, output_neurons] + hidden_neurons,
                                   [input_neurons, output_neurons] + hidden_neurons))
        al: list[list[str]] = [l for al in activation_lists for l in tails(al)]
        al.sort()
        self.activation_lists = list(tuple(l) for l, _ in groupby(al))
        il: list[list[str]] = [l for il in initialization_lists for l in tails(il)]
        il.sort()
        self.initialization_lists = list(tuple(l) for l, _ in groupby(il))

    def delta(self) -> dict[str, list[Any]]:
        return {
            "learning_rate": self.learning_rates,
            "learning_rate_feature": ["Constant"],
            "loss_feature": ["MSE", "CEL"],
            "update_feature": ["SGD"],
            "activation_feature": ["Sigmoid", "ReLu"],
            "initialization_feature": ["Glotrot_Uniform", "Glotrot_Normal", "Normal"],
            "layer": self.max_layers,
            "shape": self.shapes,
            "activation_list": self.activation_lists,
            "initialization_list": self.initialization_lists,
        }

    def gamma(self):
        return {
            "Learning_Rate": DSL()
            .Use("lr", "learning_rate")
            .In(Constructor("Learning_Rate", Literal("Constant", "learning_rate_feature") & LVar("lr"))),
            # "Learning_Rate_RDA": DSL()
            # .Use("lr", "learning_rate")
            # .In(Constructor("Learning_Rate", Literal("RDA", "learning_rate_feature") & LVar("lr"))),
            "Loss_MSE": Constructor("Loss", Literal("MSE", "loss_feature")),
            "Loss_CEL": Constructor("Loss", Literal("CEL", "loss_feature")),
            "Update_SGD": Constructor("Update", Literal("SGD", "update_feature")),
            #"Update_RDA_Momentum": Constructor("Update",
            #                                   Literal("RDA", "update_feature") &
            #                                   Literal("Momentum", "update_feature")),
            #"Update_GradientDescent": Constructor("Update", Literal("Gradient_Descent", "update_feature")),
            #"Update_GradientDescent_Momentum": Constructor("Update",
            #                                               Literal("Gradient_Descent", "update_feature") &
            #                                               Literal("Momentum", "update_feature")),
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
            "Bias_True": Constructor("Bias"),
            "Bias_False": Constructor("Bias"),
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
            .Use("al", "activation_list")
            .As(lambda af: (af,))
            .Use("wf", "initialization_feature")
            .Use("wl", "initialization_list")
            .As(lambda wf: (wf,))
            .Use("s", "shape")
            .Use("_input", Constructor("Layer") & Constructor("Dense", LVar("s") & LVar("af") & LVar("wf")))
            .In(Constructor("Model_Dense", Literal(0, "layer") & LVar("s") & LVar("al") & LVar("wl"))),
            "Network_Dense_Cons": DSL()
            .Use("af", "activation_feature")
            .Use("atl", "activation_list")
            .Use("al", "activation_list")
            .As(lambda af, atl: (af,) + atl)
            .Use("wf", "initialization_feature")
            .Use("wtl", "initialization_list")
            .Use("wl", "initialization_list")
            .As(lambda wf, wtl: (wf,) + wtl)
            .Use("m", "layer")
            .Use("n", "layer")
            .As(lambda m: m - 1)
            .Use("s1", "shape")
            .Use("s2", "shape")
            .Use("s3", "shape")
            .With(lambda s1, s2, s3: s3[0] == s1[0] and s1[1] == s2[0] and s3[1] == s2[1])
            .Use("layer", Constructor("Layer") & Constructor("Dense", LVar("s1") & LVar("af") & LVar("wf")))
            .Use("model", Constructor("Model_Dense", LVar("n") & LVar("s2") & LVar("atl") & LVar("wtl")))
            .In(Constructor("Model_Dense", LVar("m") & LVar("s3") & LVar("al") & LVar("wl"))),
            "Learner_Dense": DSL()
            .Use("n", "layer")
            .With(lambda n: n >= self.min_layers)
            .Use("s", "shape")
            .Use("lr", "learning_rate")
            .Use("lrf", "learning_rate_feature")
            .Use("lf", "loss_feature")
            .Use("uf", "update_feature")
            .Use("al", "activation_list")
            .Use("wl", "initialization_list")
            .Use("rate", Constructor("Learning_Rate", LVar("lrf") & LVar("lr")))
            .Use("loss", Constructor("Loss", LVar("lf")))
            .Use("upd", Constructor("Update", LVar("uf")))
            .Use("net", Constructor("Model_Dense", LVar("n") & LVar("s") & LVar("al") & LVar("wl")))
            .In(Constructor("Learner",
                            LVar("lrf") & LVar("lr") & LVar("lf") &
                            LVar("uf")) & Constructor("Dense", LVar("n") & LVar("s") & LVar("al") & LVar("wl"))),
        }

    @staticmethod
    def layer_dense(shape, af, wf, activation, weights, bias):
        return linear(shape, weights) >> bias(shape[1]) >> activation

    def para_lens_algebra(self):
        return {
            "Learning_Rate": (lambda n: to_para(learning_rate(n))),
            #"Learning_Rate_RDA": (lambda n: to_para(rda_learning_rate(n))),
            "Loss_MSE": Para(mse_loss),
            "Loss_CEL": Para(mse_loss),
            "Update_SGD": rda_momentum(-0.1),
            # "Update_RDA_Momentum": rda_momentum(-0.1),
            # "Update_GradientDescent": gd(-0.01),
            # "Update_GradientDescent_Momentum": momentum(-0.01, -0.1),
            "Activation_Sigmoid": to_para_init(lens.sigmoid),
            "Activation_ReLu": to_para_init(lens.relu),
            "Weights_Initial_Normal": normal(0, 0.01),
            "Weights_Initial_GlotrotUniform": glorot_uniform,
            "Weights_Initial_Glotrot": glorot_normal,
            "Bias_True": numeric_optics.para.bias,
            "Bias_False": numeric_optics.para.bias,
            "Layer_Dense": self.layer_dense,
            "Network_Dense_Start": (lambda af, al, wf, wl, s, l: l),
            "Network_Dense_Cons": (lambda af, atl, al, wf, wtl, wl, m, n, s1, s2, s3, layer, model: layer >> model),
            "Learner_Dense": (lambda n, s, lr, lrf, lf, uf, al, wl, rate, loss, upd, net:
                              (supervised_step(net, upd, loss, rate), net)),
        }
    
    def build_pytorch(self, n, s, lr, lrf, lf, uf, al, wl, rate, loss, upd, net):
        class IrisExample(nn.Module):
            def __init__(self, input_, output_):
                super(IrisExample, self).__init__()
                
                # for l, w in zip(net[0], net[1]):
                #     if isinstance(l, nn.Linear):
                #         w(l.weight.data)
                print(net)
                self.layers = net    


            def forward(self, x):
                return self.layers(x)
        model = IrisExample(s[0], s[1])
        return model, loss, upd(model.parameters(), rate)
    
    def build_linear(self, shape, af, wf, activation, weights, bias):
        layer = nn.Linear(shape[0], shape[1], bias=bias)
        weights(layer.weight.data)
        return nn.Sequential(layer, activation)
    
    #TODO: RDA is now torch.SGD, update repository!
    
    def pytorch_algebra(self):
        return {
            "Learning_Rate": (lambda n: n),
            #"Learning_Rate_RDA": (lambda n: to_para(rda_learning_rate(n))),
            "Loss_MSE": nn.MSELoss(),
            "Loss_CEL": nn.CrossEntropyLoss(),
            "Update_SGD": (lambda param, rate: torch.optim.SGD(param, lr=rate, weight_decay = 0.001, momentum = 0.9)),
            # "Update_RDA_Momentum": rda_momentum(-0.1),
            # "Update_GradientDescent": gd(-0.01),
            # "Update_GradientDescent_Momentum": momentum(-0.01, -0.1),
            "Activation_Sigmoid": nn.Sigmoid(),
            "Activation_ReLu": nn.ReLU(),
            "Weights_Initial_Normal": nn.init.normal_,
            "Weights_Initial_GlotrotUniform": nn.init.xavier_uniform_,
            "Weights_Initial_Glotrot": nn.init.xavier_normal_,
            "Bias_True": True,
            "Bias_False": False,
            "Layer_Dense": self.build_linear, #(lambda shape, af, wf, activation, weights, bias:  ([nn.Linear(shape[0], shape[1], bias=bias), activation], [weights])), # nn.Sequential(nn.Linear(shape[0], shape[1], bias=bias)).apply(weights).append(activation) ),#
            "Network_Dense_Start": (lambda af, al, wf, wl, s, l: l),
            "Network_Dense_Cons": (lambda af, atl, al, wf, wtl, wl, m, n, s1, s2, s3, layer, model: layer.extend(model)), #(layer[0] + model[0], layer[1] + model[1])), #
            "Learner_Dense": self.build_pytorch,
        }

