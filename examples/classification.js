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

let NN = new Network([
    new Layer(ReLUNeuron, 2, 16),
    new Layer(ReLUNeuron, 16, 16),
    new Layer(ReLUNeuron, 16, 1, linear=true)
]);

function shuffle(obj1, obj2) {
  var index = obj1.length;
  var rnd, tmp1, tmp2;

  while (index) {
    rnd = Math.floor(Math.random() * index);
    index -= 1;
    tmp1 = obj1[index];
    tmp2 = obj2[index];
    obj1[index] = obj1[rnd];
    obj2[index] = obj2[rnd];
    obj1[rnd] = tmp1;
    obj2[rnd] = tmp2;
  }
}

shuffle(X, y);
const num_training = 80;

// sgd
let step_size = 1.0;
for (let i = 0; i < 100; i++) {
    let idx = Math.floor(Math.random() * num_training);
    let output = NN.forward(X[idx])[0];
	
	let loss = new Scalar(1.0).sub(output.mul(y[idx])).max(new Scalar(0.0));
    NN.zero_grad();
	loss.backward();
    
	step_size = 1.0 - 0.9 * i / 100.0;
    for (let p of NN.parameters())
        p.value -= step_size * p.grad;
}

const sigmoid = x => 1.0 / (1.0 + Math.E**(-x));
let correct = 0;
for (let i = num_training; i < X.length; i++) {
	let classification = sigmoid(NN.forward(X[i])[0]) > 0.5 ? 1 : 0;
	if (classification == y[i])
		correct++;
}

console.log("Accuracy: " + correct / (X.length - num_training) + "%");

