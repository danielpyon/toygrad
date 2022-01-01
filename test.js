"use strict";

const Scalar = require("./Scalar.js");

let a = new Scalar(2.0);
let b = new Scalar(3.0);
let c = a.pow(b);
let d = b.div(c);
let e = d.pow(c);
e.backward();

console.log(a.grad);
console.log(b.grad);
