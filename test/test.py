import torch
import math
import sys

def main():
    vals = iter(map(float, sys.argv[1:]))
    TOLERANCE = 1e-6
    close_enough = lambda a, b: abs(a - b) < TOLERANCE

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    c = a ** b
    d = b / c
    e = d ** c
    f = math.e ** d
    f.backward()

    assert close_enough(a.grad.item(), next(vals))
    assert close_enough(b.grad.item(), next(vals))

    del a, b, c, d, e, f

    print('ALL TESTS PASSED')

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        main()
