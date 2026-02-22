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
        if (batch_size == 0) return error.InvalidBatchSize;

        // allocate index array [0..count)
        var indices = try allocator.alloc(usize, dataset.count);
        errdefer allocator.free(indices);

        for (indices, 0..) |*slot, i| {
            slot.* = i;
        }

        if (shuffle) {
            if (rng) |r| {
                // fisher yates shuffle
                var i = indices.len;
                while (i > 1) : (i -= 1) {
                    const j = r.uintLessThan(usize, i);
                    std.mem.swap(usize, &indices[i - 1], &indices[j]);
                }
            } else {
                // no RNG passed
                return error.MissingRngForShuffle;
            }
        }

        // allocate reusable batch buffers once
        const x = try allocator.alloc(u8, batch_size);
        errdefer allocator.free(x);

        const y = try allocator.alloc(u8, batch_size);
        errdefer allocator.free(y);

        return BatchIter{
            .dataset = dataset,
            .indices = indices,
            .pos = 0,
            .allocator = allocator,
            .batch = Batch{
                .x = x,
                .y = y,
                .n = 0,
                .capacity = batch_size,
            },
        };
    }

    pub fn deinit(self: *BatchIter) void {
        self.batch.deinit(self.allocator);
        self.allocator.free(self.indices);
        self.* = undefined;
    }

    pub fn reset(self: *BatchIter, shuffle: bool, rng: ?*std.Random) !void {
        self.pos = 0;

        for (self.indices, 0..) |*slot, i| {
            slot.* = i;
        }

        if (shuffle) {
            if (rng) |r| {
                var i = self.indices.len;
                while (i > 1) : (i -= 1) {
                    const j = r.uintLessThan(usize, i);
                    std.mem.swap(usize, &self.indices[i - 1], &self.indices[j]);
                }
            } else {
                return error.MissingRngForShuffle;
            }
        }
    }

    // Fills and returns the reusable internal batch. Returns null at end.
    pub fn next(self: *BatchIter) ?*Batch {
        if (self.pos >= self.indices.len) return null;

        const remaining = self.indices.len - self.pos;
        const n = @min(self.batch.capacity, remaining);
        self.batch.n = n;

        // Fill batch.x (normalized f32) and batch.y
        var bi: usize = 0;
        while (bi < n) : (bi += 1) {
            const ds_idx = self.indices[self.pos + bi];

            self.batch.y[bi] = self.dataset.labels[ds_idx];

            const src = self.dataset.imageBytes(ds_idx); // 784 bytes
            const dst_start = bi * MNIST_IMAGE_SIZE; // channel=1, so same as 784
            const dst = self.batch.x[dst_start .. dst_start + MNIST_IMAGE_SIZE];

            // Normalize [0,255] -> [0,1]
            for (src, 0..) |px, j| {
                dst[j] = @as(f32, @floatFromInt(px)) * (1.0 / 255.0);
            }
        }

        self.pos += n;
        return &self.batch;
    }
};
