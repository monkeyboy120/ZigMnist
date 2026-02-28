const std = @import("std");

const ConvolutionalLayer = struct {
    allocator: std.mem.Allocator,

    // hyper params
    in_c: usize,
    out_c: usize,
    k_h: usize,
    k_w: usize,
    stride: usize,
    pad: usize,

    // params
    w: []f32, // out_c * in_c * k_h * k_w
    b: []f32,

    // gradients
    dw: []f32,
    db: []f32,

    // cache for backprop
    //TODO: reference depending of lifecycle of tensor?
    cached_input: []f32, // copy of last input batch
    cached_in_shape: struct { n: usize, c: usize, h: usize, w: usize },

    // reusable output/gradient buffers
    out_buf: []f32,
    dx_buf: []f32,

    pub fn init(...) !ConvolutionalLayer {
        // TODO:
    }
    pub fn deinit(self: *ConvolutionalLayer) void {
        // TODO:
    }

    pub fn forward(self: *ConvolutionalLayer, x: Tensor4) !Tensor4 {
        // TODO:
    }

    pub fn backward(self: *ConvolutionalLayer, dout: Tensor4) !Tensor4 {
        // TODO:
    }

    pub fn zeroGrad(self: *ConvolutionalLayer) void {
        // TODO:
    }
};
