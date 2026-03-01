const std = @import("std");
const tensor = @import("tensor.zig");

pub const MNIST_IMAGE_SIZE: usize = 28 * 28;
pub const MNIST_H: usize = 28;
pub const MNIST_W: usize = 28;
pub const MNIST_C: usize = 1;

pub const DataLoaderError = error{
    InvalidMnistFile,
    UnsupportedImageShape,
    CountMismatch,
    InvalidBatchSize,
    MissingRngForShuffle,
};

pub const MnistDataset = struct {
    allocator: std.mem.Allocator,
    images: []u8,
    labels: []u8,
    count: usize,

    pub fn deinit(self: *MnistDataset) void {
        self.allocator.free(self.images);
        self.allocator.free(self.labels);
        self.* = undefined;
    }

    pub fn imageBytes(self: *const MnistDataset, idx: usize) []const u8 {
        std.debug.assert(idx < self.count);
        const start = idx * MNIST_IMAGE_SIZE;
        return self.images[start .. start + MNIST_IMAGE_SIZE];
    }

    pub fn label(self: *const MnistDataset, idx: usize) u8 {
        std.debug.assert(idx < self.count);
        return self.labels[idx];
    }

    pub fn pixel(self: *const MnistDataset, idx: usize, y: usize, x: usize) u8 {
        std.debug.assert(idx < self.count);
        std.debug.assert(y < MNIST_H);
        std.debug.assert(x < MNIST_W);
        return self.images[idx * MNIST_IMAGE_SIZE + y * MNIST_W + x];
    }
};

pub const Batch = struct {
    allocator: std.mem.Allocator,
    x: []f32,
    y: []u8,
    n: usize,
    capacity: usize,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !Batch {
        return .{
            .allocator = allocator,
            .x = try allocator.alloc(f32, capacity * MNIST_IMAGE_SIZE),
            .y = try allocator.alloc(u8, capacity),
            .n = 0,
            .capacity = capacity,
        };
    }

    pub fn deinit(self: *Batch) void {
        self.allocator.free(self.x);
        self.allocator.free(self.y);
        self.* = undefined;
    }

    pub fn xIndex(i: usize, c: usize, y: usize, x: usize) usize {
        return (((i * MNIST_C + c) * MNIST_H + y) * MNIST_W + x);
    }

    pub fn xTensor(self: *Batch) !tensor.Tensor4 {
        return tensor.Tensor4.init(self.x[0 .. self.n * MNIST_IMAGE_SIZE], self.n, MNIST_C, MNIST_H, MNIST_W);
    }

    pub fn labels(self: *const Batch) []const u8 {
        return self.y[0..self.n];
    }
};

pub const BatchIter = struct {
    dataset: *const MnistDataset,
    indices: []usize,
    pos: usize,
    batch: Batch,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        dataset: *const MnistDataset,
        batch_size: usize,
        shuffle: bool,
        rng: ?*std.Random,
    ) !BatchIter {
        if (batch_size == 0) return DataLoaderError.InvalidBatchSize;

        var indices = try allocator.alloc(usize, dataset.count);
        errdefer allocator.free(indices);

        for (indices, 0..) |*slot, i| {
            slot.* = i;
        }

        if (shuffle) try shuffleIndices(indices, rng);

        var batch = try Batch.init(allocator, batch_size);
        errdefer batch.deinit();

        return .{
            .dataset = dataset,
            .indices = indices,
            .pos = 0,
            .batch = batch,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BatchIter) void {
        self.batch.deinit();
        self.allocator.free(self.indices);
        self.* = undefined;
    }

    pub fn reset(self: *BatchIter, shuffle: bool, rng: ?*std.Random) !void {
        self.pos = 0;
        for (self.indices, 0..) |*slot, i| {
            slot.* = i;
        }
        if (shuffle) try shuffleIndices(self.indices, rng);
    }

    pub fn next(self: *BatchIter) ?*Batch {
        if (self.pos >= self.indices.len) return null;

        const remaining = self.indices.len - self.pos;
        const n = @min(self.batch.capacity, remaining);
        self.batch.n = n;

        var batch_i: usize = 0;
        while (batch_i < n) : (batch_i += 1) {
            const dataset_idx = self.indices[self.pos + batch_i];
            self.batch.y[batch_i] = self.dataset.label(dataset_idx);

            const src = self.dataset.imageBytes(dataset_idx);
            const dst_start = batch_i * MNIST_IMAGE_SIZE;
            const dst = self.batch.x[dst_start .. dst_start + MNIST_IMAGE_SIZE];

            for (src, 0..) |px, pixel_idx| {
                dst[pixel_idx] = @as(f32, @floatFromInt(px)) / 255.0;
            }
        }

        self.pos += n;
        return &self.batch;
    }
};

pub fn loadMnistDataset(
    allocator: std.mem.Allocator,
    images_path: []const u8,
    labels_path: []const u8,
) !MnistDataset {
    const image_bytes = try readDataFileAlloc(allocator, images_path);
    defer allocator.free(image_bytes);

    const label_bytes = try readDataFileAlloc(allocator, labels_path);
    defer allocator.free(label_bytes);

    if (image_bytes.len < 16 or label_bytes.len < 8) return DataLoaderError.InvalidMnistFile;

    const image_magic = readU32Be(image_bytes[0..4]);
    const image_count = readU32Be(image_bytes[4..8]);
    const rows = readU32Be(image_bytes[8..12]);
    const cols = readU32Be(image_bytes[12..16]);

    const label_magic = readU32Be(label_bytes[0..4]);
    const label_count = readU32Be(label_bytes[4..8]);

    if (image_magic != 2051 or label_magic != 2049) return DataLoaderError.InvalidMnistFile;
    if (rows != MNIST_H or cols != MNIST_W) return DataLoaderError.UnsupportedImageShape;
    if (image_count != label_count) return DataLoaderError.CountMismatch;

    const count: usize = image_count;
    const expected_image_len = 16 + count * MNIST_IMAGE_SIZE;
    const expected_label_len = 8 + count;

    if (image_bytes.len != expected_image_len or label_bytes.len != expected_label_len) {
        return DataLoaderError.InvalidMnistFile;
    }

    const images = try allocator.alloc(u8, count * MNIST_IMAGE_SIZE);
    errdefer allocator.free(images);
    @memcpy(images, image_bytes[16..]);

    const labels = try allocator.alloc(u8, count);
    errdefer allocator.free(labels);
    @memcpy(labels, label_bytes[8..]);

    return .{
        .allocator = allocator,
        .images = images,
        .labels = labels,
        .count = count,
    };
}

fn shuffleIndices(indices: []usize, rng: ?*std.Random) !void {
    const random = rng orelse return DataLoaderError.MissingRngForShuffle;
    var i = indices.len;
    while (i > 1) : (i -= 1) {
        const j = random.uintLessThan(usize, i);
        std.mem.swap(usize, &indices[i - 1], &indices[j]);
    }
}

fn readDataFileAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fs.cwd().readFileAlloc(allocator, path, std.math.maxInt(usize)) catch |err| switch (err) {
        error.FileNotFound => blk: {
            const joined = try std.fs.path.join(allocator, &.{ "data", path });
            defer allocator.free(joined);
            break :blk try std.fs.cwd().readFileAlloc(allocator, joined, std.math.maxInt(usize));
        },
        else => err,
    };
}

fn readU32Be(bytes: []const u8) u32 {
    std.debug.assert(bytes.len == 4);
    return (@as(u32, bytes[0]) << 24) |
        (@as(u32, bytes[1]) << 16) |
        (@as(u32, bytes[2]) << 8) |
        @as(u32, bytes[3]);
}
