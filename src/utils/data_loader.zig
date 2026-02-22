// store data as bytes then normalize to f32 array
const std = @import("std");

// normalize for tensors
pub const MNIST_IMAGE_SIZE: usize = 28 * 28;
pub const MNIST_H: usize = 28;
pub const MNIST_W: usize = 28;
pub const MNIST_C: usize = 1;

// basic storage as bytes
pub const MnistDataset = struct {
    allocator: std.mem.Allocator,
    images: []u8, // pack into 784 array
    labels: []u8,
    count: usize,

    pub fn deinit(self: *MnistDataset) void {
        self.allocator.free(self.images); // delete images
        self.allocator.free(self.labels); // delete labels
        self.* = undefined; // invalidate memory
    }

    // return a view into raw bytes for image 'idx'
    pub fn imageBytes(self: *const MnistDataset, idx: usize) u8 {
        std.debug.assert(idx < self.count);
        const start = idx * MNIST_IMAGE_SIZE;
        return self.images[start .. start + MNIST_IMAGE_SIZE];
    }

    // return a label for image 'idx'
    pub fn label(self: *const MnistDataset, idx: usize) u8 {
        std.debug.assert(idx < self.count);
        return self.labels[idx];
    }

    // helper for bounds checked pixel access
    pub fn pixels(self: *const MnistDataset, idx: usize, y: usize, x: usize) u8 {
        std.debug.assert(idx < self.count);
        std.debug.assert(y < MNIST_H);
        std.debug.assert(x < MNIST_W);
        return self.images[idx * MNIST_IMAGE_SIZE + y * MNIST_H + x];
    }
};

pub const Batch = struct {
    allocator: std.mem.Allocator,
    x: []f32, // tensor data [n, 1, 28, 28]
    y: []u8, // label data [n, label]
    n: usize, // actual number of samples in the batch

    capacity: usize, // capacity in samples

    pub fn deinit(self: *Batch) void {
        self.allocator.free(self.x);
        self.allocator.free(self.y);
        self.* = undefined;
    }

    // Flattened index into x for sample i, channel c, row y, col x
    pub fn xIndex(i: usize, c: usize, y: usize, x: usize) usize {
        return (((i * MNIST_C + c) * MNIST_H + y) * MNIST_W + x);
    }
};
