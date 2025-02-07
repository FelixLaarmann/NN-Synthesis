#!/usr/bin/env python

import numeric_optics.para

from itertools import product

import numeric_optics.lens as lens
from numeric_optics.para import Para, to_para, dense, linear, to_para_init
from numeric_optics.supervised import train_supervised, supervised_step, mse_loss, learning_rate, rda_learning_rate
from numeric_optics.update import gd, rda, rda_momentum, momentum
from numeric_optics.statistics import accuracy
from numeric_optics.initialize import normal, glorot_normal, glorot_uniform

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

    def __init__(self, base_repo: Linear_Repository):
        self.learning_rates = base_repo.learning_rates
        self.min_layers = base_repo.min_layers
        self.max_layers = base_repo.max_layers
        self.shapes = base_repo.shapes
        self.train_input = base_repo.train_input
        self.train_labels = base_repo.train_labels
        self.delta = base_repo.delta

    def gamma(self):
        return {
            "Experiment": DSL()
            .Use("n", "layer")
            .With(lambda n: n >= self.min_layers)
            .Use("s", "shape")
            .Use("lr", "learning_rate")
            .Use("lrf", "learning_rate_feature")
            .Use("lf", "loss_feature")
            .Use("uf", "update_feature")
            .Use("af", "activation_feature")
            .Use("wf", "initialization_feature")
            .Use("learner", Constructor("Learner",
                                        LVar("lrf") & LVar("lr") & LVar("lf") & LVar("uf") &
                                        LVar("n") & LVar("s") & LVar("af") & LVar("wf")))
            .With(lambda learner: self.test_term(learner, 0.9))
            .In(Constructor("Experiment",
                            LVar("lrf") & LVar("lr") & LVar("lf") & LVar("uf") &
                            LVar("n") & LVar("s") & LVar("af") & LVar("wf"))),
        }

    def test_term(self, term, min_accuracy):
        (step, param), model = interpret_term(term, self.para_lens_algebra())
        e_prev = None
        fwd = model.arrow.arrow.fwd
        acc = 0.0
        for e, j, i, param in train_supervised(step, param, self.train_input, self.train_labels, num_epochs=400,
                                               shuffle_data=True):
            if e == e_prev:
                continue

            e_prev = e
            predict = lambda x: fwd((param[1], x)).argmax()
            acc = accuracy(predict, self.train_input, self.train_labels.argmax(axis=1))
        return min_accuracy <= acc

    def para_lens_algebra(self):
        return {
            "Experiment": (lambda n, s, lr, lrf, lf, uf, af, wf, x: x),
        }
