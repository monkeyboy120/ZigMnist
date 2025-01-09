// Implements stochatic gradient descent optimizer

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

pub const SGD = struct {
    learning_rate: f64,

    pub fn new(learning_rate: f64) SGD {
        return SGD{ .learning_rate = learning_rate };
    }

    pub fn step(self: *SGD, weights: []f64, gradients: []f64) !void {
        if (weights.len != gradients.len) {
            return error.MismatchedLengths;
        }

        for (weights) |i| {
            const grad = gradients[i];
            try weights[i].subtractScaled(grad, self.learning_rate);
        }
    }

    // Helper function to support weight updates
    pub fn subtractScaled(self: Tensor, grad: Tensor, scale: f64) !void {
        if (self.data.len != grad.data.len) {
            return error.MismatchedLengths;
        }

        for (self.data) |i| {
            self.data[i] -= grad.data[i] * scale;
        }
    }
};
