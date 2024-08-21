#!/usr/bin/env python3

import numpy as np
import math

from experiments.dataset import load_mnist

import numeric_optics.lens as lens
from numeric_optics.para import Para, ParaInit, dense, relu, to_para
from numeric_optics.statistics import accuracy
import numeric_optics.para.convolution as image

from numeric_optics.update import rda_momentum
from numeric_optics.supervised import supervised_step, train_supervised, mse_loss, learning_rate

from clsp import (
    DSL,
    Constructor,
    LVar,
    FiniteCombinatoryLogic,
    Subtypes,
)
from clsp.types import Literal
from clsp.enumeration import interpret_term, enumerate_terms

from synthesis.linear_repository import Linear_Repository
from synthesis.convolutional_repository import Convolutional_Repository


def main() -> None:
    print(
        "NOTE: ensure you run ./get-mnist-dataset.sh to download the MNIST dataset first, otherwise this may hang trying to download it!")
    (x_train, y_train), (x_test, y_test) = load_mnist()

    base = Linear_Repository([-0.01], 5 * 5 * 5, 10,
                             [0], [],
                             [["Sigmoid", "Sigmoid", "Sigmoid"], ["ReLu", "ReLu", "ReLu"], ["ReLu", "ReLu", "Sigmoid"]],
                             [["Normal", "Normal", "Normal"]])

    conv = Convolutional_Repository(base, [5], [(3, 3), (4, 4)], (28, 28), [1, 3, 5], [(2, 2)])

    #print((base.delta() | conv.delta()))

    print("#############################\n#############################\n\n")

    target1 = Constructor("Layer") & Constructor("Correlate_2D", Literal(((13, 13, 5), (10, 10, 5)), "convolutional_shape") & Literal((4,4), "kernel_shape") & Literal("ReLu", "activation_feature") & Literal("Normal", "initialization_feature"))

    target2 = (Constructor("Model_Convolutional", Literal(1, "convolutional_layer") & Literal(((10, 10, 5), (5, 5, 5)),"convolutional_shape") & Literal(("Sigmoid",), "activation_list") & Literal(("Normal",), "initialization_list")) &
               Constructor("Model_Dense", Literal(0, "layer") & Literal((125, 10), "shape") & Literal(("Sigmoid",), "activation_list") & Literal(("Normal",), "initialization_list")))

    target3 = (Constructor("Model_Convolutional", Literal(2, "convolutional_layer") & Literal(((13, 13, 3), (5, 5, 5)), "convolutional_shape") & Literal(("ReLu", "Sigmoid",), "activation_list") & Literal(("Normal", "Normal",), "initialization_list")) &
              Constructor("Model_Dense",  Literal(0, "layer") & Literal((125, 10), "shape") & Literal(("Sigmoid",), "activation_list") & Literal(("Normal",), "initialization_list")))

    target4 = (Constructor("Model_Convolutional", Literal(3, "convolutional_layer") & Literal(((26, 26, 3), (5, 5, 5)), "convolutional_shape") & Literal(("ReLu", "Sigmoid",), "activation_list") & Literal(("Normal", "Normal",), "initialization_list")) &
               Constructor("Model_Dense", Literal(0, "layer") & Literal((125, 10), "shape") & Literal(("Sigmoid",),"activation_list") & Literal(("Normal",), "initialization_list")))

    target = (Constructor("Learner") &
              Constructor("Dense", Literal(0, "layer") & Literal((125, 10), "shape") & Literal(("Sigmoid",), "activation_list") & Literal(("Normal",), "initialization_list")) &
              Constructor("Convolutional", Literal(4, "convolutional_layer") &
                          Literal(((28, 28, 1), (5, 5, 5)), "convolutional_shape") &
                          Literal(("ReLu", "ReLu", "Sigmoid",), "activation_list") &
                          Literal(("Normal", "Normal", "Normal",), "initialization_list")))

    print(f"target: {target}")

    fcl = FiniteCombinatoryLogic((base.gamma() | conv.gamma()), Subtypes({}), (base.delta() | conv.delta()))

    #print(fcl.literals)

    results = fcl.inhabit(target)

    print("finished inhabitation")

    terms = enumerate_terms(target, results, max_count=100)

    print(f"Number of results: {len(list(enumerate_terms(target, results, max_count=100)))}")

    term_number = 1

    for t in terms:
        print("#############################\n#############################\n\n")
        print(f"Learning term {term_number}")
        print("Term: \n")
        print(t)
        print("\n")
        (step, param), model = interpret_term(t, (base.para_lens_algebra() | conv.para_lens_algebra()))
        e_prev = None
        fwd = model.arrow.arrow.fwd
        for e, j, i, param in train_supervised(step, param, x_train, y_train, num_epochs=4, shuffle_data=True):
            # only print diagnostics every 10Kth sample
            if j % 10000:
                continue

            e_prev = e
            predict = lambda x: fwd((param[1], x)).argmax()
            # NOTE: this is *TEST* accuracy, unlike iris experiment.
            acc = accuracy(predict, x_test, y_test.argmax(axis=1))
            print('epoch', e, 'sample', j, '\taccuracy {0:.4f}'.format(acc), sep='\t')

        # final accuracy
        acc = accuracy(predict, x_test, y_test.argmax(axis=1))
        print('final accuracy: {0:.4f}'.format(acc))
        term_number = term_number + 1


if __name__ == "__main__":
    main()
