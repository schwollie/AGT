# Algorithmic Game Theory (AGT) — TUM

This repository contains my implementations and experiments for the Algorithmic Game Theory course at the Technical University of Munich (TUM). The goal is to deepen my understanding of AGT concepts by coding up key algorithms and models discussed in the course.

## Contents

### Normal Form Game

The first module is a Python implementation of **Normal Form Games**. It provides:

- Representation of multi-player normal form games using utility matrices.
- Methods to enumerate action spaces and compute utilities.
- Calculation of expected utilities for mixed strategies.
- Detection of strictly dominated strategies via linear programming.

See [`normal_form_game.py`](normal_form_game.py) for the implementation and [`test_normal_form_game.py`](test_normal_form_game.py) for unit tests.

## Getting Started

1. Clone the repository.
2. Install dependencies:
    ```bash
    pip install numpy scipy
    ```
3. Run tests:
    ```bash
    pytest
    ```

## License

This project is for educational purposes.
