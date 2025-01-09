//NOTE: Implements Convolutional Layers

const std = @import("std");
const Tensor = @import("tensor").Tensor;

pub const ConvLayer = struct {
    filters: []Tensor, // Kernels
    bias: []f64, // Bias for each filter
    stride: usize, // Stride size
    padding: usize, // Padding size

    pub fn new(filters: []Tensor, bias: []f64, stride: usize, padding: usize) !ConvLayer {
        if (filters.len != bias.len) {
            return error.InvalidLayerConfiguration;
        }

        return ConvLayer{
            .filters = filters,
            .bias = bias,
            .stride = stride,
            .padding = padding,
        };
    }
};

pub fn forward(self: ConvLayer, input: Tensor, allocator: *std.mem.Allocator) !Tensor {
    const in_h = input.shape[0];
    const in_w = input.shape[1];
    const in_c = input.shape[2];

    // Check input channels match filter channels
    for (self.filters) |filter| {
        if (filter.shape[2] != in_c) {
            return error.InvalidInputShape;
        }
    }

    // Output Dimensions
    const filter_h = self.filters[0].shape[0];
    const filter_w = self.filters[0].shape[1];
    const out_h = (in_h + 2 * self.padding - filter_h) / self.stride + 1;
    const out_w = (in_w + 2 * self.padding - filter_w) / self.stride + 1;
    const out_c = self.filters.len;

    // Allocato output tensor
    var output = Tensor.new(try allocator.alloc(f64, out_h * out_w * out_c), .{ out_h, out_w, out_c });

    for (out_h) |oh| {
        for (out_w) |ow| {
            for (out_c) |oc| {
                var value: f64 = self.bias[oc];

                // Apply the filter
                const filter = self.filters[oc];
                for (filter_h) |fh| {
                    for (filter_w) |fw| {
                        for (in_c) |ic| {
                            const ih = oh * self.stride + fh - self.padding;
                            const iw = ow * self.stride + fw - self.padding;

                            if (ih >= 0 and iw >= 0 and ih < in_h and iw < in_w) {
                                const input_val = try input.at(ih, iw, ic);
                                const filter_val = try filter.at(fh, fw, ic);
                                value += input_val * filter_val;
                            }
                        }
                    }
                }

                // Store the result
                try output.set(oh, ow, oc, value);
            }
        }
    }

    return output;
}
