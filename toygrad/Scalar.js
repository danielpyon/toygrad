// Scalar for autograd
"use strict";

class Scalar {
    constructor(value) {
        this.value = value;
        this.inputs = [];
        this.backprop = () => {}; // this is the function for backpropping grads thru gates
        this.grad = 0;
    }
    add(other) {
        if (!(other instanceof Scalar))
            other = new Scalar(other);
        // forward pass
        let out = new Scalar(this.value + other.value);
        out.inputs.push(this, other);

        // given d[loss]/d[out], backprop grads to inputs with chain rule
        out.backprop = dout => {
            // add because multivariate chain rule
            this.grad += 1.0 * dout; // 1.0 is local grad: d[out]/d[grad]
            other.grad += 1.0 * dout;
        };
        return out;
    }
    sub(other) {
        return this.add(other.mul(-1.0));
    }
    mul(other) {
        if (!(other instanceof Scalar))
            other = new Scalar(other);
        let out = new Scalar(this.value * other.value);
        out.inputs.push(this, other);
        
        out.backprop = dout => {
            this.grad += other.value * dout;
            other.grad += this.value * dout;
        }
        return out;
    }
    neg() {
        return this.mul(-1.0);
    }
    div(other) {
        return this.mul(other.pow(-1.0));
    }
    pow(other) {
        if (!(other instanceof Scalar)) {
            let out = new Scalar(this.value ** other);
            out.inputs.push(this);
            out.backprop = dout => {
                let local_grad = other * this.value ** (other - 1.0);
                this.grad += local_grad * dout;
            };
            return out;
        } else {
            let out = new Scalar(this.value ** other.value);
            out.inputs.push(this, other);
            out.backprop = dout => {
                let dthis = other.value * this.value ** (other.value - 1.0) // d[out]/d[this]
                this.grad += dthis * dout;
                
                let dother = (this.value ** other.value) * Math.log(this.value); // d[out]/d[other]
                other.grad += dother * dout;
            };
            return out;
        }
    }
    exp() {
        return new Scalar(Math.E).pow(this);
    }
    sig() {
        const sigmoid = x => 1.0 / (1.0 + Math.E ** (-1.0 * x));
        let out = new Scalar(sigmoid(this.value));
        out.inputs.push(this);
        out.backprop = dout => {
            let localgrad = (1 - sigmoid(this.value)) * sigmoid(this.value);
            this.grad += localgrad * dout;
        };
        return out;
    }
    relu() {
        // relu(x) = max(0, x)
        let out = new Scalar(Math.max(0, this.value));
        out.inputs.push(this);
        out.backprop = dout => {
            this.grad += (this.value < 0.0 ? 0.0 : 1.0);
        };
        return out;
    }
    max(...others) {
        let inputs = others.concat(this); // others is a list of Scalar objects
        let values = inputs.map(x => x.value);

        // fwd pass
        const max_value = Math.max(...values);
        let max_valued_input = inputs[values.indexOf(max_value)];

        let out = new Scalar(max_value);
        out.inputs.push(...inputs);

        out.backprop = dout => {
            // only the max-valued input gets a grad of 1.0, everything else is 0.0
            max_valued_input.grad += 1.0 * dout;
        };
        return out;
    }
    
    // this function is for the last output value, to initiate backprop
    backward() {
        // d[loss]/d[loss] = 1.0 since it's the identity function
        this.grad = 1.0;

        // call backprop functions for all gates in graph
        // must go from last scalar to first
        
        let visited = new Set();
        // preorder traversal of graph
        // only visits each node once
        function call_backprop(out) {
            if (visited.has(out))
                return;
            visited.add(out);
            
            out.backprop(out.grad);
            for (let input of out.inputs) {
                call_backprop(input);
            }
        };
        call_backprop(this);
    }
}

module.exports = Scalar;
