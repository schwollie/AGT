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

    def a_is_dominated(self, a: int, i: int) -> bool:
        """
        Check if action a of player i is strictly dominated by some mixed strategy.
        Returns:
            bool: True if action a is strictly dominated, False otherwise.
        """
        num_actions = self.utility_matrix.shape[i]
        other_players = [j for j in range(self.get_num_players()) if j != i]

        # Generate all possible a_-i (actions of other players)
        A_minus_i = np.meshgrid(
            *[self.get_A_i(j) for j in other_players], indexing="ij"
        )
        A_minus_i = np.vstack([a.flatten() for a in A_minus_i]).T

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
                # Build full action profile for utility lookup
                action_profile = []
                idx_other = 0
                for j in range(self.get_num_players()):
                    if j == i:
                        action_profile.append(b_i)
                    else:
                        action_profile.append(a_minus_i[idx_other])
                        idx_other += 1
                row[b_i] = self.get_U(np.array(action_profile), i)
            row[-1] = -1  # -epsilon
            # Right-hand side: u_i(a_i, a_-i)
            action_profile = []
            idx_other = 0
            for j in range(self.get_num_players()):
                if j == i:
                    action_profile.append(a)
                else:
                    action_profile.append(a_minus_i[idx_other])
                    idx_other += 1
            rhs = self.get_U(np.array(action_profile), i)
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
