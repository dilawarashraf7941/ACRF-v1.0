"""Unit tests for `LinUCBArm` (`app/policy/linucb/arm.py`).

Covers: initial prediction, confidence bonus, update changes prediction,
deterministic behavior, mathematical correctness (verified against
explicit `numpy.linalg.inv`), and matrix dimensions.
"""

import numpy as np
import pytest

from app.context import ContextVector
from app.policy.linucb.arm import LinUCBArm, context_feature_vector
from app.policy.linucb.models import LinUCBPrediction


def _make_context(features: dict[str, float], context_id: str = "ctx-1") -> ContextVector:
    return ContextVector(
        context_id=context_id,
        features=features,
        feature_order=list(features.keys()),
        timestamp="2024-01-01T00:00:00Z",
    )


# --- context_feature_vector ---


def test_context_feature_vector_follows_feature_order() -> None:
    context = ContextVector(
        context_id="ctx-1",
        features={"b": 2.0, "a": 1.0},
        feature_order=["a", "b"],
        timestamp="2024-01-01T00:00:00Z",
    )

    x = context_feature_vector(context)

    assert x.tolist() == [1.0, 2.0]


def test_context_feature_vector_is_float64() -> None:
    x = context_feature_vector(_make_context({"a": 1.0}))
    assert x.dtype == np.float64


def test_context_feature_vector_raises_on_empty_feature_order() -> None:
    context = ContextVector(context_id="ctx-1", timestamp="2024-01-01T00:00:00Z")

    with pytest.raises(ValueError, match="feature_order is empty"):
        context_feature_vector(context)


# --- construction / validation ---


def test_matrix_dimensions() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=5)

    assert arm.A.shape == (5, 5)
    assert arm.A_inv.shape == (5, 5)
    assert arm.b.shape == (5,)


def test_initial_a_is_regularization_times_identity() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=3, regularization=2.0)

    assert np.allclose(arm.A, 2.0 * np.eye(3))
    assert np.allclose(arm.A_inv, np.eye(3) / 2.0)


def test_initial_b_is_zero() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=4)
    assert np.allclose(arm.b, np.zeros(4))


@pytest.mark.parametrize("dimension", [0, -1])
def test_invalid_dimension_raises(dimension: int) -> None:
    with pytest.raises(ValueError):
        LinUCBArm(arm_id="LogicCritic", dimension=dimension)


def test_negative_alpha_raises() -> None:
    with pytest.raises(ValueError):
        LinUCBArm(arm_id="LogicCritic", dimension=2, alpha=-0.1)


@pytest.mark.parametrize("regularization", [0.0, -1.0])
def test_non_positive_regularization_raises(regularization: float) -> None:
    with pytest.raises(ValueError):
        LinUCBArm(arm_id="LogicCritic", dimension=2, regularization=regularization)


def test_predict_raises_on_dimension_mismatch() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    context = _make_context({"a": 1.0, "b": 2.0, "c": 3.0})

    with pytest.raises(ValueError, match="dimension"):
        arm.predict(context)


def test_update_raises_on_dimension_mismatch() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    context = _make_context({"a": 1.0})

    with pytest.raises(ValueError, match="dimension"):
        arm.update(context, reward=1.0)


# --- initial prediction ---


def test_initial_prediction_expected_reward_is_zero() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    context = _make_context({"a": 1.0, "b": 2.0})

    prediction = arm.predict(context)

    assert prediction.expected_reward == 0.0


def test_initial_prediction_returns_lin_ucb_prediction() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    context = _make_context({"a": 1.0, "b": 2.0})

    prediction = arm.predict(context)

    assert isinstance(prediction, LinUCBPrediction)
    assert prediction.arm_id == "LogicCritic"
    assert prediction.context_id == "ctx-1"


# --- confidence bonus ---


def test_confidence_bonus_matches_closed_form_for_untouched_arm() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2, alpha=1.5, regularization=1.0)
    features = {"a": 3.0, "b": 4.0}
    context = _make_context(features)

    prediction = arm.predict(context)

    x = np.array([3.0, 4.0])
    expected_bonus = 1.5 * np.sqrt(x @ x)  # A_inv == I when regularization == 1.0
    assert prediction.confidence_bonus == pytest.approx(expected_bonus)
    assert prediction.upper_confidence_bound == pytest.approx(0.0 + expected_bonus)


def test_confidence_bonus_is_non_negative() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    context = _make_context({"a": -5.0, "b": 3.0})

    prediction = arm.predict(context)

    assert prediction.confidence_bonus >= 0.0


def test_confidence_bonus_scales_with_alpha() -> None:
    context = _make_context({"a": 1.0, "b": 1.0})

    low_alpha = LinUCBArm(arm_id="LogicCritic", dimension=2, alpha=0.5).predict(context)
    high_alpha = LinUCBArm(arm_id="LogicCritic", dimension=2, alpha=2.0).predict(context)

    assert high_alpha.confidence_bonus > low_alpha.confidence_bonus
    assert high_alpha.confidence_bonus == pytest.approx(4 * low_alpha.confidence_bonus)


def test_confidence_bonus_shrinks_after_repeated_updates_on_same_context() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    context = _make_context({"a": 1.0, "b": 1.0})

    bonus_before = arm.predict(context).confidence_bonus
    arm.update(context, reward=1.0)
    bonus_after = arm.predict(context).confidence_bonus

    assert bonus_after < bonus_before


# --- update changes prediction ---


def test_update_changes_expected_reward() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    context = _make_context({"a": 1.0, "b": 1.0})

    before = arm.predict(context).expected_reward
    arm.update(context, reward=1.0)
    after = arm.predict(context).expected_reward

    assert before == 0.0
    assert after != before


def test_update_with_zero_reward_does_not_change_b() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    context = _make_context({"a": 1.0, "b": 1.0})

    arm.update(context, reward=0.0)

    assert np.allclose(arm.b, np.zeros(2))


def test_update_mutates_a_a_inv_and_b_in_place() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    context = _make_context({"a": 1.0, "b": 2.0})

    a_before, a_inv_before, b_before = arm.A.copy(), arm.A_inv.copy(), arm.b.copy()
    arm.update(context, reward=1.0)

    assert not np.allclose(arm.A, a_before)
    assert not np.allclose(arm.A_inv, a_inv_before)
    assert not np.allclose(arm.b, b_before)


# --- mathematical correctness (verified against explicit inversion) ---


def test_a_inv_matches_explicit_numpy_inversion_after_one_update() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2, regularization=1.0)
    x = np.array([1.0, 2.0])
    context = _make_context({"a": 1.0, "b": 2.0})

    arm.update(context, reward=1.0)

    a_expected = np.eye(2) + np.outer(x, x)
    a_inv_expected = np.linalg.inv(a_expected)
    assert np.allclose(arm.A, a_expected)
    assert np.allclose(arm.A_inv, a_inv_expected, atol=1e-9)


def test_a_inv_matches_explicit_numpy_inversion_after_multiple_updates() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2, regularization=1.0)
    observations = [
        ({"a": 1.0, "b": 2.0}, 1.0),
        ({"a": -1.0, "b": 0.5}, 0.3),
        ({"a": 2.0, "b": -1.0}, 0.9),
    ]

    a_expected = np.eye(2)
    b_expected = np.zeros(2)
    for features, reward in observations:
        context = _make_context(features)
        arm.update(context, reward)
        x = np.array([features["a"], features["b"]])
        a_expected = a_expected + np.outer(x, x)
        b_expected = b_expected + reward * x

    a_inv_expected = np.linalg.inv(a_expected)
    assert np.allclose(arm.A, a_expected)
    assert np.allclose(arm.b, b_expected)
    assert np.allclose(arm.A_inv, a_inv_expected, atol=1e-8)


def test_expected_reward_matches_closed_form_ridge_regression() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2, regularization=1.0)
    x = np.array([1.0, 2.0])
    context = _make_context({"a": 1.0, "b": 2.0})

    arm.update(context, reward=1.0)
    prediction = arm.predict(context)

    a_expected = np.eye(2) + np.outer(x, x)
    theta_expected = np.linalg.inv(a_expected) @ x
    expected_reward_expected = float(theta_expected @ x)
    assert prediction.expected_reward == pytest.approx(expected_reward_expected)


def test_a_inv_stays_symmetric_after_many_updates() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=3)
    rng_features = [
        {"a": 1.0, "b": -2.0, "c": 0.5},
        {"a": 0.2, "b": 1.5, "c": -1.0},
        {"a": -0.7, "b": 0.3, "c": 2.0},
        {"a": 1.1, "b": 1.1, "c": 1.1},
    ]
    for features in rng_features:
        arm.update(_make_context(features), reward=0.5)

    assert np.allclose(arm.A_inv, arm.A_inv.T, atol=1e-10)


def test_a_inv_stays_positive_definite_after_many_updates() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    for i in range(20):
        context = _make_context({"a": float(i % 3), "b": float((i + 1) % 5)})
        arm.update(context, reward=float(i % 2))

    eigenvalues = np.linalg.eigvalsh(arm.A_inv)
    assert np.all(eigenvalues > 0.0)


# --- deterministic behavior ---


def test_predict_is_deterministic() -> None:
    arm = LinUCBArm(arm_id="LogicCritic", dimension=2)
    context = _make_context({"a": 1.0, "b": 2.0})

    first = arm.predict(context)
    second = arm.predict(context)

    assert first == second


def test_update_sequence_is_deterministic_across_independent_arms() -> None:
    context = _make_context({"a": 1.0, "b": 2.0})

    arm_1 = LinUCBArm(arm_id="LogicCritic", dimension=2)
    arm_2 = LinUCBArm(arm_id="LogicCritic", dimension=2)

    for arm in (arm_1, arm_2):
        arm.update(context, reward=0.7)
        arm.update(context, reward=0.2)

    assert np.array_equal(arm_1.A, arm_2.A)
    assert np.array_equal(arm_1.A_inv, arm_2.A_inv)
    assert np.array_equal(arm_1.b, arm_2.b)
