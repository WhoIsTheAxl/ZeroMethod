"""Contract tests for the pinned GymRubiksCube environment.

Run from the repository root with:
    python -m unittest discover -s tests -v

The upstream project uses the legacy Gym API, so these tests instantiate its
environment class directly instead of relying on Gym's version-sensitive
wrappers.
"""

import unittest

import numpy as np

from gym_rubiks_cube.envs.rubiksCubeEnv import RubiksCubeEnv

from zeromethod.environment.state_legality import validate_cube_state, validate_observation


INVERSE_ACTION_PAIRS = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (14, 15),
    (16, 17),
)


# Independent facelet geometry: face blocks are U, F, R, B, L, D.
_FACE_NORMALS = (
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 0),
    (0, 0, -1),
    (-1, 0, 0),
    (0, -1, 0),
)


def _facelet_geometry():
    positions = []
    normals = []
    for face, normal in enumerate(_FACE_NORMALS):
        for row in range(3):
            for column in range(3):
                if face == 0:  # U
                    position = (column - 1, 1, row - 1)
                elif face == 1:  # F
                    position = (column - 1, 1 - row, 1)
                elif face == 2:  # R
                    position = (1, 1 - row, 1 - column)
                elif face == 3:  # B
                    position = (1 - column, 1 - row, -1)
                elif face == 4:  # L
                    position = (-1, 1 - row, column - 1)
                else:  # D
                    position = (column - 1, -1, 1 - row)
                positions.append(position)
                normals.append(normal)
    return positions, normals


def _quarter_turn(vector, axis, sign):
    x, y, z = vector
    if axis == 0:
        rotated = (x, -z, y)
    elif axis == 1:
        rotated = (z, y, -x)
    else:
        rotated = (-y, x, z)
    if sign < 0:
        return tuple(-value if index != axis else value
                     for index, value in enumerate(rotated))
    return rotated


def _geometric_slice_permutation(axis, layer, sign):
    positions, normals = _facelet_geometry()
    lookup = {(position, normal): index
              for index, (position, normal) in enumerate(zip(positions, normals))}
    permutation = np.arange(54)
    for source, (position, normal) in enumerate(zip(positions, normals)):
        if position[axis] != layer:
            continue
        destination_position = _quarter_turn(position, axis, sign)
        destination_normal = _quarter_turn(normal, axis, sign)
        destination = lookup[(destination_position, destination_normal)]
        permutation[destination] = source
    return permutation


# (rotation axis: x=0/y=1/z=2, selected layer coordinate)
_ACTION_SLICES = (
    (1, 1), (1, 1), (1, 0), (1, 0), (1, -1), (1, -1),
    (2, -1), (2, -1), (2, 0), (2, 0), (2, 1), (2, 1),
    (0, -1), (0, -1), (0, 0), (0, 0), (0, 1), (0, 1),
)


class TestGymRubiksCubeEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = RubiksCubeEnv()
        self.env.scramble_params = 0
        self.env.max_steps = None
        self.initial_state = self.env.reset().copy()

    def tearDown(self):
        self.env.close()

    def assert_valid_state(self, state):
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (54,))
        self.assertTrue(np.issubdtype(state.dtype, np.integer))
        self.assertTrue(np.all((state >= 0) & (state <= 5)))
        np.testing.assert_array_equal(
            np.bincount(state, minlength=6),
            np.full(6, 9),
        )

    def test_reset_returns_solved_cube(self):
        self.assert_valid_state(self.initial_state)
        self.assertTrue(self.env.isSolved())
        self.assertEqual(self.env.action_space.n, 18)
        self.assertEqual(self.env.observation_space.shape, (54,))

    def test_every_action_is_legal_and_preserves_sticker_counts(self):
        for action in range(self.env.action_space.n):
            with self.subTest(action=action):
                self.env.reset()
                self.assertTrue(self.env.action_space.contains(action))

                observation, reward, done, info = self.env.step(action)

                self.assert_valid_state(observation)
                self.assertEqual(reward, 0)
                self.assertFalse(done)
                self.assertIsInstance(info, dict)

    def test_each_action_permutation_is_a_bijection(self):
        permutations = self.env.transform.transformation_permutations
        self.assertEqual(len(permutations), 18)

        expected = np.arange(54)
        for action, permutation in enumerate(permutations):
            with self.subTest(action=action):
                self.assertEqual(permutation.shape, (54,))
                np.testing.assert_array_equal(np.sort(permutation), expected)


    def test_every_action_is_a_geometric_quarter_turn_of_its_declared_slice(self):
        permutations = self.env.transform.transformation_permutations
        for action, (axis, layer) in enumerate(_ACTION_SLICES):
            with self.subTest(action=action, axis=axis, layer=layer):
                matches = [
                    sign for sign in (-1, 1)
                    if np.array_equal(
                        permutations[action],
                        _geometric_slice_permutation(axis, layer, sign),
                    )
                ]
                self.assertEqual(
                    len(matches), 1,
                    "action must match exactly one quarter-turn direction",
                )

    def test_inverse_action_pairs_restore_solved_state(self):
        for clockwise, counterclockwise in INVERSE_ACTION_PAIRS:
            with self.subTest(pair=(clockwise, counterclockwise)):
                self.env.reset()
                self.env.step(clockwise)
                observation, reward, done, _ = self.env.step(counterclockwise)

                np.testing.assert_array_equal(observation, self.initial_state)
                self.assertEqual(reward, 1)
                self.assertTrue(done)
                self.assertTrue(self.env.isSolved())

    def test_four_repetitions_of_each_action_restore_state(self):
        for action in range(self.env.action_space.n):
            with self.subTest(action=action):
                self.env.reset()
                result = None
                for _ in range(4):
                    result = self.env.step(action)

                observation, reward, done, _ = result
                np.testing.assert_array_equal(observation, self.initial_state)
                self.assertEqual(reward, 1)
                self.assertTrue(done)

    def test_solved_state_detection_changes_after_a_move(self):
        self.assertTrue(self.env.isSolved())
        self.env.step(0)
        self.assertFalse(self.env.isSolved())

    def test_random_move_sequence_and_its_inverse_restore_state(self):
        rng = np.random.RandomState(2026)
        actions = rng.randint(0, self.env.action_space.n, size=100)
        inverse_actions = {
            action: action + 1 if action % 2 == 0 else action - 1
            for action in range(self.env.action_space.n)
        }

        state = self.initial_state.copy()
        for action in actions:
            state = self.env.transform(state, int(action))

        self.assert_valid_state(state)

        for action in reversed(actions):
            state = self.env.transform(state, inverse_actions[int(action)])

        np.testing.assert_array_equal(state, self.initial_state)


    def test_full_action_random_walk_preserves_observation_contract(self):
        rng = np.random.RandomState(20261103)
        state = self.initial_state.copy()
        validate_observation(state)
        for step in range(1000):
            action = int(rng.randint(0, self.env.action_space.n))
            state = self.env.transform(state, action)
            with self.subTest(step=step, action=action):
                validate_observation(state)

    def test_outer_face_random_walk_passes_fixed_center_cubie_invariants(self):
        rng = np.random.RandomState(20261104)
        outer_actions = (0, 1, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17)
        state = self.initial_state.copy()
        validate_cube_state(state)
        for step in range(1000):
            action = int(rng.choice(outer_actions))
            state = self.env.transform(state, action)
            with self.subTest(step=step, action=action):
                validate_cube_state(state)

    def test_max_steps_ends_an_unsolved_episode(self):
        self.env.max_steps = 1

        observation, reward, done, info = self.env.step(0)

        self.assert_valid_state(observation)
        self.assertEqual(reward, 0)
        self.assertTrue(done)
        self.assertIsInstance(info, dict)
        self.assertFalse(self.env.isSolved())

    def test_scramble_reset_preserves_sticker_counts(self):
        self.env.scramble_params = 25
        np.random.seed(2026)

        observation = self.env.reset()

        self.assert_valid_state(observation)
        self.assertEqual(self.env.steps_since_reset, 0)


if __name__ == "__main__":
    unittest.main()
