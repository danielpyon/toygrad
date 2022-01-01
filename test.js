const Scalar = require("./Scalar.js")["Scalar"];

let a = new Scalar(2.0);
let b = new Scalar(3.0);
let c = a.mul(b);
let d = c.add(2.0);
let e = a.mul(d);
let f = e.pow(2.0);
f.backward();

console.log(a.grad);
console.log(b.grad);
