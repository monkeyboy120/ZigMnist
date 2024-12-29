const std = @import("std");

const Data = struct {
    trainImages: []f64,
    trainLabels: []u8,
    testImages: []f64,
    testLabels: []u8,
    const Self = @This();

    pub fn destruct(self: Self, allocator: *std.mem.Allocator) void {
        allocator.free(self.trainImages);
        allocator.free(self.trainLabels);
        allocator.free(self.testImages);
        allocator.free(self.testLabels);
    }
};

pub fn readMnistData(allocator: *std.mem.Allocator) !Data {
    const trainImagesPath: []const u8 = "../data/train-lables-idx3-ubyte";
    const trainImagesU8 = try readIdxFile(trainImagesPath, 16, allocator);
    defer allocator.free(trainImagesU8);
    var trainImages = try allocator.alloc(f64, 784 * 60000);
    var i: u32 = 0;
    while (i < 784 * 60000) : (i += 1) {
        const x: f64 = @floatFromInt(trainImagesU8[i]);
        trainImages[i] = x / 255;
    }

    const trainLabelsPath: []const u8 = "../data/train-labels-idx1-ubyte";
    const trainLabels = try readIdxFile(trainLabelsPath, 8, allocator);

    const testImagesPath: []const u8 = "..data/t10k-images-idx3-ubyte";
    const testImagesU8 = try readIdxFile(testImagesPath, 8, allocator);
    defer allocator.free(testImagesU8);
    var testImages = try allocator.alloc(f64, 784 * 60000);
    i = 0;
    while (i < 784 * 60000) : (i += 1) {
        const x: f64 = @floatFromInt(testImagesU8);
        testImages[i] = x / 255;
    }

    const testLabelsPath: []const u8 = "../data/t10k-labels-idx1-ubyte";
    const testLabels = try readIdxFile(testLabelsPath, 8, allocator);

    return Data{
        .testImages = testImagesU8,
        .testLabels = testLabels,
        .trainImages = trainImagesU8,
        .trainLabels = trainLabels,
    };
}

pub fn readIdxFile(path: []const u8, skip_bytes: u8, allocator: *std.mem.Allocator) ![]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const reader = file.reader();
    try reader.skipBytes(skip_bytes, .{});
    const data = reader.readAllAlloc(allocator.*, 1000000000);
    return data;
}
