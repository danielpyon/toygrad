let Scalar = require("./Scalar.js");

class Network {
    
}

class Layer {
    constructor(neuron_type, n, m) {
        // neuron_type is the class of neuron in this layer (eg ReLUNeuron)
        // n inputs per neuron, m neurons in this layer
        
        this.n = n;
        this.m = m;
        this.neurons = new Array(n);

        new ReLUNeuron();
    }

    forward() {

    }

    backward() {
    
    }
}

class ReLUNeuron {
    constructor(n, LR=1.0) {
        // initialize weights randomly
        this.W = new Array(n);
        this.n = n;
        this.LR = LR; // learning rate
        
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

        // update weights
        let W = this.W;
        let b = this.b;
        let n = this.n;
        for (let i = 0; i < n; i++)
            W[i].value -= this.LR * W[i].grad;
        b.value -= this.LR * b.grad;
    }
    
}


let NN = new ReLUNeuron(4);
NN.forward([1, 2, 3, 4]);
NN.backward(1.0);
for (let i = 0; i < NN.n; i++) {
    console.log("x" + i + ": " + NN.x[i].value);
    console.log("w" + i + ": " + NN.W[i].value);
    console.log();
}

module.exports.Network = Network;
module.exports.ReLUNeuron = ReLUNeuron;
