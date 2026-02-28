const std = @import("std");
// define different errors we could have
pub const TensorError = error{
    ShapeMismatch,
    OutOfBounds,
    NonContiguousSize,
};

//NOTE: Tensor 2 is matrix view (2d), Tensor 4 is image batch view (4d)

pub const Tensor2 = struct {
    data: []f32,
    n: usize, // num of rows
    d: usize, // num of cols

    pub fn init(data: []f32, n: usize, d: usize) !Tensor2 {
        // check if dim is good first
        if (data.len != n * d) return TensorError.ShapeMismatch;
        return .{ .data = data, .n = n, .d = d };
    }

    pub fn len(self: *const Tensor2) usize {
        return self.data.len;
    }

    pub fn idx(self: *const Tensor2, i: usize, j: usize) usize {
        std.debug.assert(i < self.n);
        std.debug.assert(j < self.d);
        return i * self.d + j;
    }

    pub fn get(self: *const Tensor2, i: usize, j: usize) f32 {
        return self.data[self.idx(i, j)];
    }

    pub fn set(self: *Tensor2, i: usize, j: usize, in: f32) void {
        const k = self.idx(i, j);
        self.data[k] = in;
    }

    pub fn row(self: *const Tensor2, i: usize) []const f32 {
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
        for (self.data) |*x| x.* = value;
    }

    pub fn asConst(self: *const Tensor2) Tensor2Const {
        return .{
            .data = self.data,
            .n = self.n,
            .d = self.d,
        };
    }

    pub fn reshapeTo4(self: *Tensor2, c: usize, h: usize, w: usize) !Tensor4 {
        if (self.d != c * w * h) return TensorError.ShapeMismatch;
        return Tensor4.init(self.data, self.n, c, h, w);
    }
};

pub const Tensor2Const = struct {
    data: []const f32,
    n: usize, // num of rows
    d: usize, // num of cols

    pub fn init(data: []const f32, n: usize, d: usize) !Tensor2Const {
        if (data.len != n * d) return TensorError.ShapeMismatch;
        return .{
            .data = data,
            .n = n,
            .d = d,
        };
    }
};

const Tensor4 = struct {
    data: []f32,
    n: usize, // batch
    c: usize, // channels
    h: usize, // height
    w: usize, // width

    pub fn init(data: []f32, n: usize, c: usize, h: usize, w: usize) !Tensor4 {
        if (data.len != c * w * h) return TensorError.ShapeMismatch;
        return .{
            .data = data,
            .n = n,
            .c = c,
            .h = h,
            .w = w,
        };
    }

    pub fn deinit(self: *Tensor4) void {}
};
