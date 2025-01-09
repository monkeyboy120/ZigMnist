//NOTE: Pooling layer for max and average pooling

const std = @import("std");
const Tensor = @import("tensor").Tensor;

pub const PoolLayer = struct {
    pool_size: usize,
    stride: usize,

    pub fn new(pool_size: usize, stride: usize) !PoolLayer {
        return PoolLayer{
            .pool_size = pool_size,
            .stride = stride,
        };
    }
};

pub fn maxPool(self: PoolLayer, input: Tensor, allocator: *std.mem.Allocator) !Tensor {
    const in_h = input.shape[0];
    const in_w = input.shape[1];
    const in_c = input.shape[2];

    // Output dimensions
    const out_h = (in_h - self.pool_size) / self.stride + 1;
    const out_w = (in_w - self.pool_size) / self.stride + 1;

    // Allocato output tensor
    var output = Tensor.new(try allocator.alloc(f64, out_h * out_w * in_c), .{ out_h, out_w, in_c });

    // Max Pooling
    for (out_h) |oh| {
        for (out_w) |ow| {
            for (in_c) |ic| {
                var max_value = -std.math.inf(f64);

                for (self.pool_size) |ph| {
                    for (self.pool_size) |pw| {
                        const ih = oh * self.stride + ph;
                        const iw = ow * self.stride + pw;

                        if (ih < in_h and iw < in_w) {
                            const value = try input.at(ih, iw, ic);
                            if (value > max_value) {
                                max_value = value;
                            }
                        }
                    }
                }

                // Set the maximum output tensor
                try output.set(oh, ow, ic, max_value);
            }
        }
    }
    return output;
}

//TODO: averagePool function
