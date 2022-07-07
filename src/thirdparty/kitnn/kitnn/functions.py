from torch.autograd import Function


class StablePow(Function):

    def __init__(self, power):
        self.power = power

    def forward(self, input):
        self.save_for_backward(input)
        return input.clamp(min=0).pow(self.power)

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        eps = 0.00001
        grad_input[input <= eps] = eps
        grad_input[input > eps] = \
            self.power * grad_input[input > eps].pow(self.power - 1)
        return grad_input
