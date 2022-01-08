let Scalar = require("./Scalar.js");

class Module {
    parameters() {}
    zero_grad() {}
    forward(x) {}
    backward(dout) {}
}

// Network: contains layers
class Network extends Module {
    constructor(layers) {
        super();
        this.modules = layers;
    }
    parameters() {
        let params = [];
        for (let module of this.modules)
            params.push(...module.parameters());
        return params;
    }
    zero_grad() {
        for (let module of this.modules)
            module.zero_grad();
    }
    forward(x) {
        let out = x;
        for (let i = 0; i < this.modules.length; i++)
            out = this.modules[i].forward(out);
        return out;
    }
    backward(dout) {
        this.modules[this.modules.length - 1].backward(dout);
    }
}

// Layer: contains neurons
class Layer extends Module {
    constructor(NeuronType, n, m, linear=false) {
        // NeuronType is the class of neuron in this layer (eg ReLUNeuron)
        // n inputs per neuron, m neurons in this layer
        super();

        this.n = n;
        this.m = m;
        this.modules = [];

        for (let i = 0; i < m; i++)
            this.modules.push(new NeuronType(n, linear));
    }

    parameters() {
        // Return list of all parameters of neurons in this layer
        let params = [];
        for (let i = 0; i < this.m; i++)
            params.push(...this.modules[i].parameters());
        return params;
    }

    zero_grad() {
        for (let i = 0; i < this.m; i++)
            this.modules[i].zero_grad();
    }

    forward(x) {
        // Returns [m x 1] vector of each output value in layer
        let out = [];
        for (let i = 0; i < this.m; i++)
            out.push(this.modules[i].forward(x));
        return out;
    }

    backward(dout) {
        for (let i = 0; i < this.m; i++)
            this.modules[i].backward(dout[i]);
    }
}

// Neuron: contains weights
class ReLUNeuron extends Module {
    constructor(n, linear=false) {
        super();

        this.linear = linear;

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
    parameters() {
        // return [w0, w1, ... wn, b]
        let params = [];
        for (let weight of this.W)
            params.push(weight);
        params.push(this.b);
        return params;
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

        let out = sum.add(this.b);
        if (this.linear)
            out = out.relu();
        
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

module.exports.Network = Network;
module.exports.Layer = Layer;
module.exports.ReLUNeuron = ReLUNeuron;
