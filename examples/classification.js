// Softmax linear classifier

let Scalar = require("../toygrad/Scalar.js");
const fs = require("fs");

let dataset = JSON.parse(fs.readFileSync("./spiral_dataset.json"));

let X = dataset["X"];
let y = dataset["y"];

const N = X.length; // 100 data points
const D = X[0].length; // 2 input dimensions
const K = Math.max(...y); // 3 output classes

