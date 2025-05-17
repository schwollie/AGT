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
