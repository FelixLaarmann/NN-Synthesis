from typing import Any
from clsp import (
    DSL,
    Constructor,
    LVar,
    FiniteCombinatoryLogic,
    Subtypes,
)
from clsp.types import Literal
from clsp.enumeration import interpret_term, enumerate_terms

class Base_Repository:

    def __init__(self, learning_rates: list[float],
                 dimensions: list[int], max_hidden: int):
        self.learning_rates = learning_rates
        self.dimensions = dimensions
        self.max_hidden = max_hidden

    def delta(self) -> dict[str, list[Any]]:
        return {
            "learning_rate": self.learning_rates,
            "dimension": self.dimensions,
            "hidden": list(range(0, self.max_hidden + 1, 1)),
        }

    def gamma(self):
        return {
            "Layer": DSL()
            .Use("n", "dimension")
            .Use("bias", Constructor("Bias", LVar("n")))
            .Use("af", Constructor("activation_function"))
            .In(Constructor("layer", LVar("n"))),
            "Bias": DSL()
            .Use("n", "dimension")
            .In(Constructor("Bias", LVar("n"))),
            "ReLu": Constructor("activation_function"),
            "Model": DSL()
            .Use("in", "dimension")
            .Use("out", "dimension")
            .Use("l", Constructor("layer", LVar("out")))
            .In(Constructor("model", Constructor("input", LVar("in")) & Constructor("output", LVar("out"))) & Constructor("hidden", Literal(0, "hidden"))),
            "Model_cons": DSL()
            .Use("in", "dimension")
            .Use("out", "dimension")
            .Use("neurons", "dimension")
            .Use("n", "hidden")
            .Use("m", "hidden")
            .As(lambda n: n-1)
            .Use("l", Constructor("layer", LVar("neurons")))
            .Use("m", Constructor("model", Constructor("input", LVar("neurons")) & Constructor("output", LVar("out"))) & Constructor("hidden", LVar("m")))
            .In(Constructor("model", Constructor("input", LVar("in")) & Constructor("output", LVar("out"))) & Constructor("hidden", LVar("n"))),
            "System": DSL()
            .Use("in", "dimension")
            .Use("out", "dimension")
            .Use("lr", "learning_rate")
            .Use("n", "hidden")
            .Use("m", Constructor("model", Constructor("input", LVar("in")) & Constructor("output", LVar("out"))) & Constructor("hidden", LVar("n")))
            .Use("l", Constructor("loss"))
            .In(Constructor("system", Constructor("input", LVar("in")) & Constructor("output", LVar("out"))) & LVar("lr") & Constructor("hidden", LVar("n"))),
            "Loss": Constructor("loss"),
        }


repo = Base_Repository([-0.01], [2,3,4,8,10], 5)

print(repo.delta())

print("#############################\n#############################\n\n")

target = Constructor("system", Constructor("input", Literal(4, "dimension")) & Constructor("output", Literal(3, "dimension"))) & Constructor("hidden", Literal(3, "hidden"))


print(f"target: {target}")

fcl = FiniteCombinatoryLogic(repo.gamma(), Subtypes({}), repo.delta())

results = fcl.inhabit(target)

terms = enumerate_terms(target, results, max_count=100000)

print(f"Number of results: {len(list(enumerate_terms(target, results, max_count=100000)))}")

print(list(terms)[0])
