// Connects everything into neural network

const std = @import("std");
const Tensor = @import("tensor.zig").tensor;
const FCLayer = @import("fc.zig").FCLayer;

pub const NN = struct {
    layers: []FCLayer,

    pub fn new(layers: []FCLayer) !NN {
        return NN{
            .layers = layers,
        };
    }

    pub fn forward(self: NN, input: Tensor, allocator: *std.mem.Allocator) !Tensor {
        var output = input;
        for (self.layers) |layer| {
            output = try layer.forward(output, allocator);
        }
        return output;
    }
};
