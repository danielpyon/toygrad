class Network {
    
}

class ReLUNeuron {
    constructor(W, b) {
        this.W = W; // [n x 1] array of Scalar
        this.b = b;
    }
    forward(x) {
        // given x: [n x 1]
        // return ReLU(W@x + b)
        
        this.x = x; // store x
        
        let product = new Scalar(0.0);
        for (let i = 0; i < W.length; i++) {
            product = product.add(W[i].mul(x[i]));
        }

        let out = product.add(this.b).relu();
        this.out = out; // store out for backward pass

        return out;
    }
    backward(dout) {
        // given dL/dout
        // backpropagate the gradient to inputs
        this.out.backprop(dout);
    }
    
}

module.export.Network = Network;
module export.ReLUNeuron = ReLUNeuron;
