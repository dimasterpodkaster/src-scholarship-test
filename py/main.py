from pathlib import Path
from sparse_matrix import SparseMatrix
import numpy as np

from sparse_matrix import SparseMatrix

test_dir = Path(__file__).parent.parent / 'data'

tolerance = 1e-2


def compare(m1: SparseMatrix, m2: SparseMatrix):
    diff = m1.to_dense() - m2.to_dense()
    return np.max(abs(diff))


def check_power_2():
    for fname in test_dir.iterdir():
        if not fname.suffix == '.in':
            continue
        print(f'Test {fname.name} ... ', end='')
        m = SparseMatrix(file=fname)
        print(SparseMatrix.to_dense(m))
        val, col, poin = SparseMatrix.to_massive(m)
        print("Values", val)
        print("Columns", col)
        print("Pointers", poin)
        m_power_2 = m @ m

        answer = SparseMatrix(file=test_dir / f'{fname.stem}.out-2')
        print(SparseMatrix.to_dense(m))
        err = compare(m_power_2, answer)

        if err < tolerance:
            print(f'success, error: {err:.3g}.')
        else:
            print(f'failed, error: {err:.3g}.')


def check_power_5():
    power = 5
    for fname in test_dir.iterdir():
        if not fname.suffix == '.in':
            continue
        print(f'Test {fname.name} ... ', end='')
        m = SparseMatrix(file=fname)
        m_power_p = m ** power

        answer = SparseMatrix(file=test_dir / f'{fname.stem}.out-5')
        err = compare(m_power_p, answer)

        if err < tolerance:
            print(f'success, error: {err:.3g}.')
        else:
            print(f'failed, error: {err:.3g}.')


def check_power_999():
    power = 999
    fname_in = test_dir / '3.in'
    print(f'Test {fname_in.name} ... ', end='')
    m = SparseMatrix(file=fname_in)
    m_power_p = m ** power

    answer = SparseMatrix(file=test_dir / f'3.out-999')
    err = compare(m_power_p, answer)

    if err < tolerance:
        print(f'success, error: {err:.3g}.')
    else:
        print(f'failed, error: {err:.3g}.')


if __name__ == '__main__':
    print('Checking power 2')
    check_power_2()
    print('Checking power 5')
    check_power_5()
    print('Checking power 999')
    check_power_999()
