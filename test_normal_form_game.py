import numpy as np

from normal_form_game import NormalFormGame


def provide_test_game_matrix() -> np.ndarray:
    matrix = {
        "T": {"L": (3, 1), "R": (0, 0)},
        "M": {"L": (0, 3), "R": (3, 2)},
        "B": {"L": (1, 1), "R": (1, 2)},
    }

    numpy_matrix = np.zeros((3, 2, 2))
    for i, row in enumerate(matrix):
        for j, col in enumerate(matrix[row]):
            numpy_matrix[i][j] = matrix[row][col]

    return numpy_matrix


def test_get_num_players():
    matrix = provide_test_game_matrix()
    game = NormalFormGame(matrix)
    assert game.get_num_players() == 2


def test_get_A_i():
    matrix = provide_test_game_matrix()
    game = NormalFormGame(matrix)
    np.testing.assert_array_equal(game.get_A_i(0), np.array([0, 1, 2]))
    np.testing.assert_array_equal(game.get_A_i(1), np.array([0, 1]))


def test_compute_A():
    matrix = provide_test_game_matrix()
    game = NormalFormGame(matrix)
    expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
    np.testing.assert_array_equal(game.A, expected)


def test_get_U_all_players():
    matrix = provide_test_game_matrix()
    game = NormalFormGame(matrix)
    # (row 0, col 0) -> (3, 1)
    np.testing.assert_array_equal(game.get_U(np.array([0, 0])), np.array([3, 1]))
    # (row 1, col 1) -> (3, 2)
    np.testing.assert_array_equal(game.get_U(np.array([1, 1])), np.array([3, 2]))
    # (row 2, col 0) -> (1, 1)
    np.testing.assert_array_equal(game.get_U(np.array([2, 0])), np.array([1, 1]))


def test_get_U_single_player():
    matrix = provide_test_game_matrix()
    game = NormalFormGame(matrix)
    # (row 0, col 0) -> player 0: 3, player 1: 1
    assert game.get_U(np.array([0, 0]), 0) == 3
    assert game.get_U(np.array([0, 0]), 1) == 1
    # (row 1, col 1) -> player 0: 3, player 1: 2
    assert game.get_U(np.array([1, 1]), 0) == 3
    assert game.get_U(np.array([1, 1]), 1) == 2
    # (row 2, col 1) -> player 0: 1, player 1: 2
    assert game.get_U(np.array([2, 1]), 0) == 1
    assert game.get_U(np.array([2, 1]), 1) == 2


def test_init_sets_utility_matrix_and_A():
    matrix = provide_test_game_matrix()
    game = NormalFormGame(matrix)
    np.testing.assert_array_equal(game.utility_matrix, matrix)
    np.testing.assert_array_equal(game.A, game.compute_A())


def test_A_except_i():
    matrix = provide_test_game_matrix()
    game = NormalFormGame(matrix)
    # Test for player 0
    expected = np.array([[0, -1], [1, -1], [2, -1]])
    np.testing.assert_array_equal(game.A_except_i(1), expected)

    # Testt for player 1
    expected = np.array([[1, 0], [1, 1]])
    np.testing.assert_array_equal(game.A_except_i(0, replace=1), expected)


def test_get_expected_utility_2by2():
    matrix_2by2 = np.array([[[2, 2], [0, 0]], [[0, 0], [1, 1]]])
    game = NormalFormGame(matrix_2by2)
    s = [np.array([0.5, 0.5]), np.array([0.25, 0.75])]
    e = game.get_e_u(s)
    np.testing.assert_allclose(e, [0.625, 0.625])

    # own simple example
    matrix_2by2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    game = NormalFormGame(matrix_2by2)
    s = [np.array([1, 0]), np.array([0.25, 0.75])]
    e = game.get_e_u(s)
    np.testing.assert_allclose(e, [2.5, 3.5])


def test_a_is_dominated():
    matrix_2by2 = np.array([[[2, 2], [0, 3]], [[3, 0], [1, 1]]])
    game = NormalFormGame(matrix_2by2)
    assert game.a_is_dominated(0, 0)
    assert game.a_is_dominated(0, 1)
    assert not game.a_is_dominated(1, 0)
    assert not game.a_is_dominated(1, 1)


def test_a_is_dominated_2():
    matrix_2by2 = np.array([[[3, 1], [0, 0]], [[0, 3], [3, 2]], [[1, 1], [1, 2]]])
    game = NormalFormGame(matrix_2by2)
    assert game.a_is_dominated(2, 0)
    assert not game.a_is_dominated(0, 0)
    assert not game.a_is_dominated(1, 0, excluded_a=np.array([[0, 0]]))


def test_it_dominated_reasonable_actions():
    matrix_2by2 = np.array([[[3, 1], [0, 0]], [[0, 3], [3, 2]], [[1, 1], [1, 2]]])
    game = NormalFormGame(matrix_2by2)
    result = game.get_iterative_dominated_actions()

    np.testing.assert_array_equal(result[0], np.array([0, 2]))
    np.testing.assert_array_equal(result[1], np.array([1, 1]))
    np.testing.assert_array_equal(result[2], np.array([0, 1]))

    # check reasonable actions, which are not dominated
    reasonable = game.get_reasonable_actions()
    np.testing.assert_array_equal(reasonable[0], np.array([0, 0]))
    np.testing.assert_array_equal(reasonable[1], np.array([1, 0]))


def test_it_dominated_actions2():
    matrix_2by2 = np.array([[[1, 3], [1, 0]], [[0, 0], [2, 1]], [[3, 1], [0, 3]]])
    game = NormalFormGame(matrix_2by2)
    result = game.get_iterative_dominated_actions()
    print(result)


def test_maxmin():
    matrix_2by2 = np.array([[[2, 1], [0, 0]], [[0, 0], [1, 2]]])
    game = NormalFormGame(matrix_2by2)
    maxmin0 = game.maximin(0)
    maxmin1 = game.maximin(1)

    # check if the maxmin is correct
    np.testing.assert_array_equal(maxmin0[1], np.array([1 / 3, 2 / 3]))
    np.testing.assert_array_equal(maxmin1[1], np.array([2 / 3, 1 / 3]))


def test_maxmin2():
    # R=0, P=1, S=2, L=3 (Indices for Rock, Paper, Scissors, Lava)
    # Player 1 is the row player, Player 2 is the column player.
    # Payoffs are (Player 1's utility, Player 2's utility).
    # Win = +1, Lose = -1, Draw = 0.
    # Standard Rock-Paper-Scissors rules apply.
    # Lava beats Rock, Paper, and Scissors. Lava vs Lava is a draw.

    rpsl_matrix = np.array(
        [
            # Opponent (Player 2) plays:
            # Rock    Paper     Scissors  Lava
            [(0, 0), (-1, 1), (1, -1), (-1, 1)],  # Player 1 plays Rock
            [(1, -1), (0, 0), (-1, 1), (-1, 1)],  # Player 1 plays Paper
            [(-1, 1), (1, -1), (0, 0), (-1, 1)],  # Player 1 plays Scissors
            [(1, -1), (1, -1), (1, -1), (0, 0)],  # Player 1 plays Lava
        ]
    )

    game = NormalFormGame(rpsl_matrix)

    # Calculate maximin for Player 0 (Player 1 in description)
    maxmin0_value, maxmin0_strategy = game.maximin(0)

    # Calculate maximin for Player 1 (Player 2 in description)
    maxmin1_value, maxmin1_strategy = game.maximin(1)

    # For this game, the maximin strategy for both players is to play Lava.
    # This guarantees a minimum payoff of 0.
    # The strategy vector [P(Rock), P(Paper), P(Scissors), P(Lava)]
    # will be [0, 0, 0, 1].
    expected_maximin_value = 0.0
    expected_strategy = np.array([0.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(maxmin0_value, expected_maximin_value, atol=1e-7)
    np.testing.assert_allclose(maxmin0_strategy, expected_strategy, atol=1e-7)

    np.testing.assert_allclose(maxmin1_value, expected_maximin_value, atol=1e-7)
    np.testing.assert_allclose(maxmin1_strategy, expected_strategy, atol=1e-7)
