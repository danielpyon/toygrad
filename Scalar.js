// Scalar for autograd
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
    neg(other) {
        return this.mul(-1.0);
    }
    pow(other) {
        if (!(other instanceof Scalar)) {
            let out = new Scalar(this.value ** other.value);
            out.inputs.push(this);
            out.backprop = dout => {
                let local_grad = other.value * this.value ** (other.value - 1.0);
                this.grad += local_grad * dout;
            };
            return out;
        } else {

        }
    }
    
    // this function is for the last output value, to initiate backprop
    backward() {
        // d[loss]/d[last output] = 1.0 since it's the identity function
        this.grad = 1.0;

        // call backprop functions for all gates in graph
        // must go from last scalar to first
        function call_backprop(out) {
            // preorder
            out.backprop(out.grad);
            for (let input of out.inputs) {
                call_backprop(input);
            }
        };
        call_backprop(this);
    }
}

exports.Scalar = Scalar;
