const Scalar = require("../toygrad/Scalar.js");

let a = new Scalar(2.0);
let b = new Scalar(3.0);
let c = a.pow(b);
let d = b.div(c);
let e = d.pow(c);
let f = d.exp();
f.backward();
console.log(a.grad);
console.log(b.grad);

a = new Scalar(3.4);
b = new Scalar(-1.3);
f = a.add(b.sig()).div(a.sig().add(a.add(b).pow(2.0)));
f.backward()

console.log(a.grad);
console.log(b.grad);

a = new Scalar(3.0);
b = new Scalar(-4.0);
c = new Scalar(2.0);
d = new Scalar(-1.0);
e = a.mul(b);
f = c.max(d);
g = e.add(f);
let h = g.mul(2.0);
h.backward();
console.log(a.grad);
console.log(b.grad);
console.log(c.grad);
console.log(d.grad);
