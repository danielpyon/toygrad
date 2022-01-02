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

