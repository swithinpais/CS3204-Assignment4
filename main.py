import timeit
import random
import logging
import sys

number = int | float


def matrix_multiplication(A: list[list[number]], B: list[list[number]]) -> list[list[number]]:
    """Multiplies two matrices, A * B. Matrices must be of compatible shape.

    Args:
        A (list[list[number]]): The first matrix (n * m).
        B (list[list[number]]): The second matrix (m * p).

    Raises:
        ValueError: Raised if either A or B is empty.
        ValueError: Raised if A and B have incompatible shape.

    Returns:
        list[list[number]]: The resultant matrix (n * p)
    """
    if not A or not B or not A[0] or not B[0]:
        raise ValueError("Matrices cannot be empty")

    if len(A[0]) != len(B):
        raise ValueError("Matrices dimensions are not compatible")

    n = len(A)
    m = len(B)
    p = len(B[0])

    C = [[0 for _ in range(p)] for _ in range(n)]

    for i in range(n):
        for k in range(m):
            for j in range(p):
                C[i][j] += A[i][k] * B[k][j]

    return C


def create_random_square_matrix(n: int, a: int, b: int) -> list[list[int]]:
    """Creates a random square matrix with integer entries in the range [a, b]

    Args:
        n (int): The size of the matrix
        a (int): Start of the range (Inclusive)
        b (int): End of the range (Inclusive)

    Returns:
        list[list[int]]: The random n x n matrix
    """
    return [[random.randint(a, b) for _ in range(n)] for _ in range(n)]


def set_logging():
    if "--debug" in sys.argv:
        logging.basicConfig(level=logging.DEBUG)
        return
    logging.basicConfig(level=logging.INFO)


def get_times(ns: list[int], A: int, B: int, number: int = 1000) -> list[float]:
    """Runs the matrix multiplication algorithm with given parameters.

    Args:
        ns (list[int]): A list of the sizes, n, to use.
        A (int): The start of the range (Inclusive).
        B (int): The end of the range (Inclusive).
        number (int, optional): The number of times to run the function, for each n. Defaults to 1000.

    Returns:
        list[float]: The resulting times.
    """

    times = []
    for n in ns:
        logging.debug(f"Running function for {n = }")
        M1 = create_random_square_matrix(n, A, B)
        M2 = create_random_square_matrix(n, A, B)

        glbls = {"matrix_multiplication": matrix_multiplication,
                 "M1": M1, "M2": M2}
        t = timeit.timeit("matrix_multiplication(M1, M2)",
                          globals=glbls, number=number)
        times.append(t)

    return times


def main() -> None:

    set_logging()

    A, B = 0, 100
    random.seed(0)

    ns = [2, 3, 5, 10, 50, 100, 200, 500, 1000, 1500, 2000]

    times = get_times(ns, A, B)

    logging.info("-"*12 + "TIMES" + "-"*12)
    for (n, t) in zip(ns, times, strict=True):

        logging.info(f"Size {n} * {n} took:\t{t:.5}")


if __name__ == "__main__":
    main()
