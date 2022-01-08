// Softmax linear classifier

let Scalar = require("../toygrad/Scalar.js");
let Network = require("../toygrad/Network.js")["Network"];
let ReLUNeuron = require("../toygrad/Network.js")["ReLUNeuron"];
let Layer = require("../toygrad/Network.js")["Layer"];

const fs = require("fs");

let dataset = JSON.parse(fs.readFileSync("./spiral_dataset.json"));

let X = dataset["X"];
let y = dataset["y"];

// spiral dataset contains 3 classes but we're only doing binary classification
// so we need to remove a class
X.splice(200, 100);
y.splice(200, 100);

const N = X.length; // 200 data points
const D = X[0].length; // 2 input dimensions
const K = Math.max(...y); // 2 output classes

let NN = new Network([
    new Layer(ReLUNeuron, D, 16), // D = 2
    new Layer(ReLUNeuron, 16, 16),
    new Layer(ReLUNeuron, 16, 1, linear=true)
]);

let hinge = actual, guess => Math.max(0, 1 - actual * guess);

function loss(batch_size) {
    // use X and y, split into buckets of batch_size size
    // do fwd pass with NN, calc hinge loss
    // return the loss (Scalar value)
}

/*
training loop will look something like this:
for 100 iterations
    l = loss(20)
    
    NN.zero_grad()
    l.backward()

    for each parameter p
        p -= stepsize * p.grad
*/
