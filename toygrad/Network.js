let Scalar = require("./Scalar.js");

// Each one of these is a "Module", so that we can save code in the "backward" call

class Module {
    zero_grad() {
        for (let i = 0; i < this.modules.length; i++)
            this.modules[i].zero_grad();
    }
    // Every module has "modules"
    backward(dout) {
        // dout is [m x 1], same as this.modules
        for (let i = 0; i < this.modules.length; i++)
            this.modules[i].backward(dout[i]);
    }
}

// Network: contains layers
class Network extends Module {
    constructor(layers) {
        super();
        
        // layers is a list of Layer objects
        this.modules = layers;
    }
    forward(x) {
        let out = x;
        for (let i = 0; i < this.modules.length; i++)
            out = this.modules[i].forward(out);
        return out;
    }
    backward(dout) {
        // dout is wrt last layer
        this.modules[this.modules.length - 1].backward(dout);
    }
}

// Layer: contains fully connected neurons
class Layer extends Module {
    constructor(NeuronType, n, m) {
        // NeuronType is the class of neuron in this layer (eg ReLUNeuron)
        // n inputs per neuron, m neurons in this layer
        super();

        this.n = n;
        this.m = m;
        this.modules = [];

        for (let i = 0; i < m; i++)
            this.modules.push(new NeuronType(n));
    }

    forward(x) {
        // Returns [m x 1] vector of each output value in layer
        let out = [];
        for (let i = 0; i < this.m; i++)
            out.push(this.modules[i].forward(x));
        return out;
    }
}

// Neuron: contains weights
class ReLUNeuron extends Module {
    constructor(n, LR=1.0) {
        super();

        // initialize weights randomly
        this.W = new Array(n);
        this.n = n;
        
        // Standard Normal variate using Box-Muller transform.
        function randn_bm() {
            var u = 0, v = 0;
            while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
            while(v === 0) v = Math.random();
            return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
        }

        for (let i = 0; i < this.W.length; i++)
            this.W[i] = new Scalar(randn_bm());

        this.b = new Scalar(0.0);
    }
    zero_grad() {
        for (let i = 0; i < this.n; i++)
            this.W[i].grad = 0.0;
        this.b.grad = 0.0;
    }
    forward(x) {
        // given x: [n x 1]
        // return ReLU(W@x + b)
        
        let n = this.n;
        for (let i = 0; i < n; i++)
            if (!(x[i] instanceof Scalar))
                x[i] = new Scalar(x[i]);

        this.x = x; // store x
        
        let products = new Array(n);
        for (let i = 0; i < n; i++)
            products[i] = this.W[i].mul(x[i]);
        
        let sum = new Scalar(0.0);
        for (let i = 0; i < n; i++)
            sum = sum.add(products[i]);

        let out = sum.add(this.b).relu();
        this.out = out; // store out for backward pass

        return out;
    }
    backward(dout) {
        // given dL/dout
        // backpropagate the gradient to inputs
        
        this.grad = dout;
        let visited = new Set();
        function call_backprop(out) {
            if (visited.has(out))
                return;
            visited.add(out);
            
            out.backprop(out.grad);
            for (let input of out.inputs) {
                call_backprop(input);
            }
        };
        call_backprop(this.out);
    }
    
}

/*
let NN = new ReLUNeuron(4);
NN.forward([1, 2, 3, 4]);
NN.backward(1.0);
for (let i = 0; i < NN.n; i++) {
    console.log("x" + i + ": " + NN.x[i].value);
    console.log("w" + i + ": " + NN.W[i].value);
    console.log();
}
*/

/*
let NN = new Layer(ReLUNeuron, 5, 3);
NN.forward([1, 2, 3, 4, 5]);
NN.backward([0.1, 0.2, 0.3, 0.4, 0.5]);
console.log(NN.neurons)
console.log(NN.neurons[0].W[0].grad);
*/

let NN = new Network([
    new Layer(4, 3),
    new Layer(3, 2),
    new Layer(2, 1)
]);

// figure out how to softmax the last layer
NN.forward([1, 2, 3, 4]);
NN.backward([1.0]);

console.log(NN.modules[0]);

module.exports.Network = Network;
module.exports.Layer = Layer;
module.exports.ReLUNeuron = ReLUNeuron;
