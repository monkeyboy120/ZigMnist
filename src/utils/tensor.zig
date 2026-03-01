const std = @import("std");

pub const TensorError = error{
    ShapeMismatch,
    OutOfBounds,
};

pub const Tensor2 = struct {
    data: []f32,
    n: usize,
    d: usize,

    pub fn init(data: []f32, n: usize, d: usize) !Tensor2 {
        if (data.len != n * d) return TensorError.ShapeMismatch;
        return .{ .data = data, .n = n, .d = d };
    }

    pub fn len(self: Tensor2) usize {
        return self.data.len;
    }

    pub fn idx(self: Tensor2, i: usize, j: usize) usize {
        std.debug.assert(i < self.n);
        std.debug.assert(j < self.d);
        return i * self.d + j;
    }

    pub fn get(self: Tensor2, i: usize, j: usize) f32 {
        return self.data[self.idx(i, j)];
    }

    pub fn set(self: *Tensor2, i: usize, j: usize, value: f32) void {
        self.data[self.idx(i, j)] = value;
    }

    pub fn row(self: Tensor2, i: usize) []const f32 {
        std.debug.assert(i < self.n);
        const start = i * self.d;
        return self.data[start .. start + self.d];
    }

    pub fn rowMut(self: *Tensor2, i: usize) []f32 {
        std.debug.assert(i < self.n);
        const start = i * self.d;
        return self.data[start .. start + self.d];
    }

    pub fn fill(self: *Tensor2, value: f32) void {
        @memset(self.data, value);
    }

    pub fn asConst(self: Tensor2) Tensor2Const {
        return .{
            .data = self.data,
            .n = self.n,
            .d = self.d,
        };
    }

    pub fn reshapeTo4(self: Tensor2, c: usize, h: usize, w: usize) !Tensor4 {
        if (self.d != c * h * w) return TensorError.ShapeMismatch;
        return Tensor4.init(self.data, self.n, c, h, w);
    }
};

pub const Tensor2Const = struct {
    data: []const f32,
    n: usize,
    d: usize,

    pub fn init(data: []const f32, n: usize, d: usize) !Tensor2Const {
        if (data.len != n * d) return TensorError.ShapeMismatch;
        return .{
            .data = data,
            .n = n,
            .d = d,
        };
    }
};

pub const Tensor4 = struct {
    data: []f32,
    n: usize,
    c: usize,
    h: usize,
    w: usize,

    pub fn init(data: []f32, n: usize, c: usize, h: usize, w: usize) !Tensor4 {
        if (data.len != n * c * h * w) return TensorError.ShapeMismatch;
        return .{
            .data = data,
            .n = n,
            .c = c,
            .h = h,
            .w = w,
        };
    }

    pub fn len(self: Tensor4) usize {
        return self.data.len;
    }

    pub fn idx(self: Tensor4, sample: usize, channel: usize, y: usize, x: usize) usize {
        std.debug.assert(sample < self.n);
        std.debug.assert(channel < self.c);
        std.debug.assert(y < self.h);
        std.debug.assert(x < self.w);
        return (((sample * self.c + channel) * self.h + y) * self.w + x);
    }

    pub fn get(self: Tensor4, sample: usize, channel: usize, y: usize, x: usize) f32 {
        return self.data[self.idx(sample, channel, y, x)];
    }

    pub fn set(self: *Tensor4, sample: usize, channel: usize, y: usize, x: usize, value: f32) void {
        self.data[self.idx(sample, channel, y, x)] = value;
    }

    pub fn fill(self: *Tensor4, value: f32) void {
        @memset(self.data, value);
    }

    pub fn flatten(self: Tensor4) !Tensor2 {
        return Tensor2.init(self.data, self.n, self.c * self.h * self.w);
    }

    pub fn asConst(self: Tensor4) Tensor4Const {
        return .{
            .data = self.data,
            .n = self.n,
            .c = self.c,
            .h = self.h,
            .w = self.w,
        };
    }
};

pub const Tensor4Const = struct {
    data: []const f32,
    n: usize,
    c: usize,
    h: usize,
    w: usize,

    pub fn init(data: []const f32, n: usize, c: usize, h: usize, w: usize) !Tensor4Const {
        if (data.len != n * c * h * w) return TensorError.ShapeMismatch;
        return .{
            .data = data,
            .n = n,
            .c = c,
            .h = h,
            .w = w,
        };
    }
};
