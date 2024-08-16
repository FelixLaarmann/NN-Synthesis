#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import argparse
from synthesis.linear_repository import Linear_Repository

from experiments.dataset import load_iris

import numeric_optics.lens as lens
from numeric_optics.para import Para, to_para, dense, linear
from numeric_optics.supervised import train_supervised, supervised_step, mse_loss, learning_rate
from numeric_optics.update import gd, rda
from numeric_optics.statistics import accuracy

from clsp import (
    DSL,
    Constructor,
    LVar,
    FiniteCombinatoryLogic,
    Subtypes,
)
from clsp.types import Literal
from clsp.enumeration import interpret_term, enumerate_terms

def main(iris_data) -> None:
    # Load data from CSV
    train_input, train_labels = load_iris(iris_data)

    base = Linear_Repository([-0.01], 4, 3,
                             [0, 1, 2], [*range(15, 25, 1)],
                             [["Sigmoid", "Sigmoid", "Sigmoid"], ["ReLu", "ReLu", "ReLu"]],
                             [["Normal", "Normal", "Normal"]])

    print(base.delta())

    print("#############################\n#############################\n\n")

    target = Constructor("Learner") & Constructor("Dense", Literal((4, 3), "shape") & Literal(0, "layer"))

    print(f"target: {target}")

    fcl = FiniteCombinatoryLogic(base.gamma(), Subtypes({}), base.delta())

    results = fcl.inhabit(target)

    terms = enumerate_terms(target, results, max_count=100000)

    print(f"Number of results: {len(list(enumerate_terms(target, results, max_count=100000)))}")

    term_number = 1

    for t in terms:
        print("#############################\n#############################\n\n")
        print(f"Learning term {term_number}")
        print("Term: \n")
        print(t)
        print("\n")
        (step, param), model = interpret_term(t, base.para_lens_algebra())
        e_prev = None
        fwd = model.arrow.arrow.fwd
        for e, j, i, param in train_supervised(step, param, train_input, train_labels, num_epochs=400,
                                               shuffle_data=True):
            # print accuracy diagnostic every epoch
            if e == e_prev:
                continue

            e_prev = e
            predict = lambda x: fwd((param[1], x)).argmax()
            acc = accuracy(predict, train_input, train_labels.argmax(axis=1))
            print('epoch', e + 1, '\ttraining accuracy {0:.4f}'.format(acc), end='\r')
        print('epoch', e + 1, '\ttraining accuracy {0:.4f}'.format(acc))
        term_number = term_number + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iris-data', default='data/iris.csv')
    #parser.add_argument('model', choices=['linear', 'dense', 'hidden'])
    args = parser.parse_args()
    main(args.iris_data)

