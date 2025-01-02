const layer = @import("layer.zig");
const nll = @import("nll.zig");
const mnist = @import("mnist.zig");
const relu = @import("relu.zig");

const std = @import("std");

const INPUT_SIZE: u32 = 784;
const OUTPUT_SIZE: u32 = 10;
const BATCH_SIZE: u32 = 32;
const EPOCHS: u32 = 25;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const data = try mnist.readMnistData(&allocator);
    defer data.destruct(allocator);

    const firstImage = data.trainImages[0..784];
    const firstLabel = data.trainLabels[0];

    std.debug.print("First Label: {d}\n", .{firstLabel});
    std.debug.print("First Image: {d}\n", .{firstImage});
}
