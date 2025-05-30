from itertools import product
from typing import Tuple

import numpy as np
from scipy.optimize import linprog


class NormalFormGame:
    def __init__(self, utility_matrix: np.ndarray):
        """
        A class representing a normal form game.
        Args:
            utility_matrix (np.ndarray): A N-D numpy array representing the utility matrix of the game.
                The first n-1 dimensions represents the number of ordered actions.
                The last dimensions is the utility of the players.
                For example, a 3D matrix with shape (3, 2, 2) represents a game with
                3 actions for player 1 and 2 actions for player 2.
        """
        self.utility_matrix = utility_matrix  # utility matrix of the game
        self.A = self.compute_A()  # The action space of the game

    def get_num_players(self) -> int:
        """
        Get the number of players in the game.
        Returns:
            int: The number of players.
        """
        return self.utility_matrix.shape[-1]

    def get_A_i(self, i: int) -> np.ndarray:
        """
        Get the actions of the game for a specific player
        Args:
            i (int): The player index
        Returns:
            np.ndarray: The complete list of actions for player i e.g (0,1,2) if player i has 3 actions.
        """
        return np.arange(self.utility_matrix.shape[i])

    def compute_A(self) -> np.ndarray:
        """
        Get the cartesian product of actions for all players.
        Returns:
            np.ndarray: A list of actions for each player with shape (,n) e.g (..., (3, 6), ...) where player 1 plays action 3 and player 2 plays action 6.
        """
        A = np.meshgrid(
            *[self.get_A_i(i) for i in range(self.get_num_players())], indexing="ij"
        )
        return np.vstack([a.flatten() for a in A]).T

    def A_except_i(self, i: int, replace: int = -1) -> np.ndarray:
        """
        Get the cartesian product of actions for all players except player i. Where action of player i is padded with -1.
        Args:
            i (int): The player index
            replace (int): The action to be replaced with, defaults to -1.
        Returns:
            np.ndarray: The complete list of actions for all other players e.g (..., (0, -1, 2), ...) where i = 1, padded with -1 and the rest of other players possible actions.
        """
        assert (
            replace < self.utility_matrix.shape[i]
        ), f"replace action must be less than max action '{self.utility_matrix.shape[i]}' for player '{i}' "
        x = self.A[self.A[:, i] == 0]
        x[:, i] = replace
        return x

    def get_U(self, a: np.ndarray, i: int = -1) -> np.ndarray:
        """
        Get the utility of the game for a specific set of actions.
        Args:
            a (np.ndarray): an action profile for each player e.g (1, 0, 2) where player one plays action 1, player 2 plays action 0 and player 3 plays action 2.
            i (int): The player index. If -1, return the utility for all players.
        Returns:
            np.ndarray: The utility of the game for the given actions. In shape (n,) where n is the number of players. of just the utility of the player as int if i != -1
        """
        if i == -1:
            return self.utility_matrix[tuple(a)]
        else:
            return self.utility_matrix[tuple(a)][i]

    def get_e_u(self, s: np.ndarray) -> np.ndarray:
        """
        Get the expected utility of the game for a specific strategy profile.
        Args:
            s (np.ndarray): strategy in form of a probability distribution over the action space for each player. e.g ((0.6, 0.4), (1)) Game with 2 players, player 1 has 2 actions and player 2 has 1 action. Player 1 plays action 0 with probability 0.6 and action 1 with probability 0.4, player 2 plays action 1 with probability 1.
        Returns:
            np.ndarray: The expected utility of the game for each player for the given strategy profile.
        """
        p = np.prod([s[i][self.A[:, i]] for i in range(self.get_num_players())], axis=0)
        u = self.utility_matrix[tuple(self.A.T)]
        return (p[:, None] * u).sum(axis=0)

    def a_is_dominated(
        self, a: int, i: int, excluded_a: np.ndarray = np.ndarray(0)
    ) -> bool:
        """
        Check if action a of player i is strictly dominated by some mixed strategy.
        Args:
            a (int): The action to check.
            i (int): The player index.
            excluded_a (np.ndarray): (Actions for each player to exclude from the check. Defaults to an empty array. Has shape (e, 2) with ((player_id, excluded_action_id),...).
        Returns:
            bool: True if action a is strictly dominated, False otherwise.
        """
        num_actions = self.utility_matrix.shape[i]

        # exclude actions, important for iterative domination
        A_minus_i = self.A_except_i(i)
        if excluded_a.size > 0:
            for excl in excluded_a:
                A_minus_i = A_minus_i[A_minus_i[:, excl[0]] != excl[1]]

        A_minus_i = np.delete(A_minus_i, i, 1)

        # Variables: s_i(b_i) for each b_i in A_i, and epsilon
        # So total num_actions + 1 variables
        c = np.zeros(num_actions + 1)
        c[-1] = -1  # maximize epsilon <=> minimize -epsilon

        # Constraints
        A_ub = []
        b_ub = []

        # For each a_-i, build the constraint
        for a_minus_i in A_minus_i:
            # For each b_i in A_i, build the action profile (b_i, a_-i)
            row = np.zeros(num_actions + 1)
            for b_i in range(num_actions):
                action_profile = np.insert(a_minus_i, i, b_i)
                row[b_i] = self.get_U(np.array(action_profile), i)
            row[-1] = -1  # -epsilon

            # Right-hand side: u_i(a_i, a_-i)
            action_profile = np.insert(a_minus_i, i, a)
            rhs = self.get_U(action_profile, i)

            # Constraint: sum_b s_i(b) * u_i(b, a_-i) - epsilon >= u_i(a, a_-i)
            # <=> sum_b s_i(b) * u_i(b, a_-i) - epsilon - u_i(a, a_-i) >= 0
            # <=> -sum_b s_i(b) * u_i(b, a_-i) + epsilon <= -u_i(a, a_-i)
            A_ub.append(-row)
            b_ub.append(-rhs)

        # Probability constraints: sum_b s_i(b) = 1
        A_eq = [np.zeros(num_actions + 1)]
        A_eq[0][:num_actions] = 1
        b_eq = [1]

        # Bounds: s_i(b) >= 0, epsilon >= 0
        bounds = [(0, 1)] * num_actions + [(0, None)]

        # Solve LP
        res = linprog(
            c,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            A_eq=np.array(A_eq),
            b_eq=np.array(b_eq),
            bounds=bounds,
            method="highs",
        )

        if res.success and res.x[-1] > 1e-8:
            return True  # strictly dominated
        else:
            return False  # not strictly dominated

    def get_iterative_dominated_actions(self) -> np.ndarray:
        """
        Get all dominated actions of the game by iteratively applying domination.
        Returns:
            np.ndarray: All dominated actions of the game.
        """
        dominated_actions = np.ndarray((0, 2), dtype=int)  # ((player_id, action_id))
        undominated_actions = np.ndarray((0, 2), dtype=int)
        for i in range(self.get_num_players()):
            for a in self.get_A_i(i):
                undominated_actions = np.append(undominated_actions, [[i, a]], axis=0)

        last_size = undominated_actions.shape[0] + 1
        while last_size > undominated_actions.shape[0]:
            last_size = undominated_actions.shape[0]
            to_remove = np.ndarray((0, 2), dtype=int)
            for x in undominated_actions:
                if self.a_is_dominated(x[1], x[0], dominated_actions):
                    to_remove = np.append(to_remove, [x], axis=0)
                    dominated_actions = np.append(dominated_actions, [x], axis=0)

            # delete all to_remove from undominated_actions
            if len(to_remove) > 0:
                # Remove all rows in undominated_actions that are present in to_remove
                mask = ~np.any(
                    np.all(undominated_actions[:, None] == to_remove, axis=2), axis=1
                )
                undominated_actions = undominated_actions[mask]

        return dominated_actions

    def get_reasonable_actions(self) -> np.ndarray:
        """
        Get all reasonable actions of the game by iteratively applying domination.
        Returns:
            np.ndarray: All reasonable actions of the game.
        """
        dominated_actions = self.get_iterative_dominated_actions()
        reasonable_actions = np.ndarray((0, 2), dtype=int)
        for i in range(self.get_num_players()):
            for a in self.get_A_i(i):
                if not np.any(np.all(dominated_actions == [i, a], axis=1)):
                    reasonable_actions = np.append(reasonable_actions, [[i, a]], axis=0)
        return reasonable_actions

    def maximin(self, i: int) -> Tuple[float, np.ndarray]:
        """
        Calculate maximin strategy for the game for player i
        Args:
            i (int): The player index.
        Returns:
            Tuple[float, np.ndarray]: The security level and one maximin strategy for player i as an array of propabilities for the actions of player i
        """
        num_players = self.get_num_players()
        player_i_num_actions = self.utility_matrix.shape[i]

        # LP variables: s_i(0), ..., s_i(player_i_num_actions-1), U*_i
        # Total variables: player_i_num_actions + 1
        # Objective: Maximize U*_i, which is equivalent to minimizing -U*_i
        # c is the coefficient vector for the objective function.
        c = np.zeros(player_i_num_actions + 1)
        c[-1] = -1  # Coefficient for U*_i (we minimize -U*_i)

        # Inequality constraints (A_ub * x <= b_ub):
        # sum_{b_k} s_i(b_k) * u_i(b_k, a_-i) >= U*_i  for all a_-i
        # This is rewritten as: U*_i - sum_{b_k} s_i(b_k) * u_i(b_k, a_-i) <= 0
        A_ub_list = []
        b_ub_list = []  # All RHS for these constraints will be 0

        # Determine opponent action profiles (a_-i)
        opponent_action_sets = []
        original_opponent_indices = (
            []
        )  # To map product indices to original player indices
        for p_idx in range(num_players):
            if p_idx != i:
                opponent_action_sets.append(self.get_A_i(p_idx))
                original_opponent_indices.append(p_idx)

        if (
            not opponent_action_sets
        ):  # This is a single-player game (player i is the only player)
            # Constraint becomes: sum_{b_k} s_i(b_k) * u_i(b_k) >= U*_i
            # (where u_i(b_k) is utility of player i for their own action b_k)
            # Rewritten: U*_i - sum_{b_k} s_i(b_k) * u_i(b_k) <= 0
            row = np.zeros(player_i_num_actions + 1)
            for k_action_idx in range(player_i_num_actions):
                # Construct action profile for player i (only player)
                action_profile = np.array([k_action_idx], dtype=int)
                utility_val = self.get_U(action_profile, i)
                row[k_action_idx] = -utility_val  # Coefficient for s_i(k_action_idx)
            row[-1] = 1  # Coefficient for U*_i
            A_ub_list.append(row)
            b_ub_list.append(0)
        else:  # Multi-player game
            # Iterate over all combinations of opponents' actions (a_-i)
            for opponent_actions_tuple in product(*opponent_action_sets):
                row = np.zeros(player_i_num_actions + 1)
                current_full_profile = np.zeros(num_players, dtype=int)

                # Populate opponent actions in the full profile
                for opponent_tuple_idx, original_p_idx in enumerate(
                    original_opponent_indices
                ):
                    current_full_profile[original_p_idx] = opponent_actions_tuple[
                        opponent_tuple_idx
                    ]

                # For each action b_k of player i, get u_i(b_k, a_-i)
                for k_action_idx in range(player_i_num_actions):
                    current_full_profile[i] = k_action_idx  # Set player i's action
                    utility_val = self.get_U(current_full_profile, i)
                    row[k_action_idx] = (
                        -utility_val
                    )  # Coefficient for s_i(k_action_idx)

                row[-1] = 1  # Coefficient for U*_i
                A_ub_list.append(row)
                b_ub_list.append(0)

        A_ub = (
            np.array(A_ub_list) if A_ub_list else None
        )  # Handle case with no inequality constraints if necessary (should not happen here)
        b_ub = np.array(b_ub_list) if b_ub_list else None

        # Equality constraints (A_eq * x == b_eq):
        # sum_{b_k} s_i(b_k) = 1
        A_eq = np.zeros((1, player_i_num_actions + 1))
        A_eq[0, :player_i_num_actions] = 1  # Coefficients for s_i(b_k) are 1
        b_eq = np.array([1])  # RHS is 1

        # Bounds for variables:
        # 0 <= s_i(b_k) <= 1 for probabilities
        # U*_i is unbounded (None, None)
        bounds = [(0, 1)] * player_i_num_actions + [(None, None)]

        # Solve the linear program
        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
        )

        if res.success:
            maximin_strategy_player_i = res.x[:player_i_num_actions]
            # Security level U*_i is the value of the objective function.
            # Since we minimized -U*_i, the optimal value res.fun = -U*_i.
            # So, U*_i = -res.fun. Alternatively, U*_i is the last variable in res.x.
            security_level = res.x[-1]  # Or -res.fun, should be equivalent

            # Ensure probabilities are non-negative and sum to 1 (due to potential floating point issues)
            maximin_strategy_player_i = np.maximum(
                0, maximin_strategy_player_i
            )  # clamp to 0
            maximin_strategy_player_i /= np.sum(
                maximin_strategy_player_i
            )  # re-normalize

            return security_level, maximin_strategy_player_i
        else:
            # This should ideally not happen for well-posed maximin problems.
            raise ValueError(
                f"Linear program for maximin strategy for player {i} did not solve: {res.message}"
            )
