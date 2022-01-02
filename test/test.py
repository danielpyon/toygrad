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

    # f(x, y) = (x + sigmoid(y)) / (sigmoid(x) + (x + y)^2)
    a = torch.tensor(3.4, requires_grad=True)
    b = torch.tensor(-1.3, requires_grad=True)
    sigmoid = lambda x: 1.0 / (1.0 + math.e ** (-x))
    f = (a + sigmoid(b)) / (sigmoid(a) + (a + b)**2)
    f.backward()
    
    assert close_enough(a.grad.item(), next(vals))
    assert close_enough(b.grad.item(), next(vals))
    del a, b, f

    a = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(-4.0, requires_grad=True)
    c = torch.tensor(2.0, requires_grad=True)
    d = torch.tensor(-1.0, requires_grad=True)
    e = a * b
    f = torch.max(c, d)
    g = e + f
    h = g * 2.0
    h.backward()

    assert close_enough(a.grad.item(), next(vals))
    assert close_enough(b.grad.item(), next(vals))
    assert close_enough(c.grad.item(), next(vals))
    assert close_enough(d.grad.item(), next(vals))
    del a, b, c, d, e, f, g, h

    print('ALL TESTS PASSED')

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        main()
