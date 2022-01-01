import torch

if __name__ == '__main__':
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    c = a ** b
    d = b / c
    e = d ** c
    e.backward()

    print(a.grad)
    print(b.grad)
