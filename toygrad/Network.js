let Scalar = require("./Scalar.js");

class Network {
    
}

class ReLUNeuron {
    constructor(n) {
        // initialize weights randomly
        this.W = new Array(n);
        
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
        
        for (let i = 0; i < x.length; i++)
            if (!(x[i] instanceof Scalar))
                x[i] = new Scalar(x[i]);

        this.x = x; // store x
        
        let product = new Scalar(0.0);
        for (let i = 0; i < this.W.length; i++) {
            product = product.add(this.W[i].mul(x[i]));
        }

        let out = product.add(this.b).relu();
        this.out = out; // store out for backward pass

        return out;
    }
    backward(dout) {
        // given dL/dout
        // backpropagate the gradient to inputs
        this.out.backward();
    }
    
}


let NN = new ReLUNeuron(4);
NN.forward([1, 2, 3, 4]);
NN.backward(1.0);

console.log(NN.x[0].grad);
console.log(NN.W[0].grad);

module.exports.Network = Network;
module.exports.ReLUNeuron = ReLUNeuron;
