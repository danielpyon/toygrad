const Scalar = require("./Scalar.js")["Scalar"];

let a = new Scalar(2.0);
let b = new Scalar(3.0);
c = a.mul(b);
d = c.add(2.0);
e = a.mul(d);
f = e.pow(2.0);
f.backward();

console.log(a.grad);
console.log(b.grad);
