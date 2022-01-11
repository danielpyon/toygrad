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

const hinge = actual, guess => Math.max(0, 1 - actual * guess);
const step_size = 1.0;

function get_random_batch(batch_size) {
    const random_int = max => Math.floor(Math.random() * max);
    let random_indices = [];
    for (let i = 0; i < batch_size; i++)
        random_indices.push(random_int(X.length));
    return random_indices;
}

for (let i = 0; i < 100; i++) {
    let indices = get_random_batch(20);
    let outputs = [];
    for (let idx of indices)
        outputs.push(NN.forward(X[idx]));

    NN.zero_grad();
    
    for (let p of NN.parameters())
        p.value -= step_size * p.grad;
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
