//! By convention, main.zig is where your main function lives in the case that
//! you are building an executable. If you are making a library, the convention
//! is to delete this file and start with root.zig instead.
const std = @import("std");
const data_loader = @import("utils/data_loader.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Pretend you've loaded MNIST files into these:
    var ds = try loadMnistDataset(allocator, "train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    defer ds.deinit();

    var prng = std.Random.DefaultPrng.init(12345);
    var rng = prng.random();

    var it = try BatchIter.init(allocator, &ds, 64, true, &rng);
    defer it.deinit();

    while (it.next()) |batch| {
        // batch.x shape: [batch.n, 1, 28, 28] packed in NCHW
        // batch.y shape: [batch.n]
        // feed to your CNN forward pass
        // cnn.forward(batch.x, batch.n);
    }

    // Next epoch:
    try it.reset(true, &rng);
}
