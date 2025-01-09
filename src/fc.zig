// Implements fully connected layers
const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

pub const FCLayer = struct {
    weights: Tensor,
    bias: Tensor,

    pub fn new(input_size: usize, output_size: usize, allocator: *std.mem.Allocator) !FCLayer {
        const weights = try Tensor.random(.{ input_size, output_size, 1 }, allocator);
        const bias = try Tensor.zeros(.{ 1, output_size, 1 }, allocator);
        return FCLayer{
            .weights = weights,
            .bias = bias,
        };
    }

    pub fn forward(self: FCLayer, input: Tensor, allocator: *std.mem.Allocator) !Tensor {
        if (input.shape[1] != self.weights.shape[0] or input.shape[2] != self.weights.shape[2]) {
            return error.InvalidInputShape;
        }

        var output = try Tensor.matmul(input, self.weights, allocator);
        defer allocator.free(output.data);
        return try output.add(self.bias, allocator);
    }
};
