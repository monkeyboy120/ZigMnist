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
    const trainImagesPath: []const u8 = "/Users/joshuala/Projects/2024/mnistzig/data/train-images.idx3-ubyte";
    const trainLabelsPath: []const u8 = "/Users/joshuala/Projects/2024/mnistzig/data/train-labels.idx1-ubyte";
    const testImagesPath: []const u8 = "/Users/joshuala/Projects/2024/mnistzig/data/t10k-images.idx3-ubyte";
    const testLabelsPath: []const u8 = "/Users/joshuala/Projects/2024/mnistzig/data/t10k-labels.idx1-ubyte";

    const trainImagesU8 = try readIdxFile(trainImagesPath, 16, allocator);
    defer allocator.free(trainImagesU8);
    var trainImages = try allocator.alloc(f64, trainImagesU8.len);
    var i: u32 = 0;
    while (i < trainImagesU8.len) : (i += 1) {
        const x: f64 = @floatFromInt(trainImagesU8[i]);
        trainImages[i] = x / 255;
    }

    const trainLabels = try readIdxFile(trainLabelsPath, 8, allocator);

    const testImagesU8 = try readIdxFile(testImagesPath, 8, allocator);
    defer allocator.free(testImagesU8);
    var testImages = try allocator.alloc(f64, testImagesU8.len);
    i = 0;
    while (i < testImagesU8.len) : (i += 1) {
        const x: f64 = @floatFromInt(testImagesU8[i]);
        testImages[i] = x / 255;
    }

    const testLabels = try readIdxFile(testLabelsPath, 8, allocator);

    return Data{
        .testImages = testImages,
        .testLabels = testLabels,
        .trainImages = trainImages,
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
