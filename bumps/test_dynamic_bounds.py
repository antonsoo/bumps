"""Dynamic fit coordinates must be ready when the fit problem is constructed."""

import numpy as np

from bumps.curve import Curve
from bumps.fitproblem import FitProblem
from bumps.parameter import Parameter
from bumps.serialize import deserialize, serialize


def line(x, m, b):
    return m * x + b


def make_model():
    x = np.arange(1.0, 7.0)
    return Curve(line, x, 2 * x + 0.1, np.ones_like(x), m=2.0, b=1.0)


def test_dynamic_coordinate_available_on_construction():
    model = make_model()
    model.m.range(0, 4)
    model.b.range(0, model.m)
    problem = FitProblem(model)
    assert problem.labels() == ["b value01", "m"]
    initial = problem.getp().copy()
    ids = [p.id for p in problem.parameters]
    problem.model_reset()
    assert problem.labels() == ["b value01", "m"]
    assert ids == [p.id for p in problem.parameters]
    np.testing.assert_array_equal(problem.getp(), initial)


def test_hidden_dynamic_dependencies_initialized():
    model = make_model()
    model.m.range(0, 4)
    root = Parameter(4.0, name="root").range(2, 8)
    top = Parameter(2.0, name="top").range(0, root)
    model.b.range(0, top)
    problem = FitProblem(model)
    assert problem.labels() == ["b value01", "top value01", "root", "m"]
    assert np.isfinite(problem.nllf())
    values = problem.getp().copy()
    problem.model_reset()
    np.testing.assert_array_equal(problem.getp(), values)


def test_reset_rebinds_dynamic_mapping():
    model = make_model()
    model.m.range(0, 4)
    model.b.range(0, model.m)
    problem = FitProblem(model)
    coordinate = model.b.slot.value01
    model.b.range(0, 2 * model.m)
    problem.model_reset()
    assert model.b.slot.value01 is coordinate
    assert model.b.slot.prior.limits == model.b.prior.limits == (0, 4)
    assert model.b.value == 2.0


def test_dynamic_bound_constructor_retains_dependency():
    upper = Parameter(2.0, name="upper").range(1, 4)
    bounded = Parameter(1.0, name="bounded", bounds=(0, upper))
    assert bounded.bounds[1] is upper


def test_serialized_dynamic_fit_retains_coordinates_and_bounds():
    model = make_model()
    model.m.range(0, 4)
    model.b.range(0, model.m)
    problem = FitProblem(model)
    restored = deserialize(serialize(problem))
    copy = restored._models[0]
    assert not model.b.fixed
    assert not copy.b.fixed
    assert restored.labels() == problem.labels() == ["b value01", "m"]
    assert [p.id for p in restored.parameters] == [p.id for p in problem.parameters]
    np.testing.assert_array_equal(restored.getp(), problem.getp())
    assert copy.b.bounds[1] is copy.m
    copy.m.value = 3.0
    assert copy.b.value == 1.5
    assert copy.b.prior.limits == (0, 3)
