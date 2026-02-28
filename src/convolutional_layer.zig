const std = @import("std");

const TensorError = error{ ShapeMismatch };

pub const ConvolutionalLayer = struct {
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
    b: []f32, // out_c

    // gradients
    dw: []f32,
    db: []f32,

    // cache for backprop
    cached_input: []f32, // copy of last input batch: N * in_c * H * W
    cached_in_shape: struct { n: usize, c: usize, h: usize, w: usize },

    // reusable output/gradient buffers (owned by layer)
    out_buf: []f32, // N * out_c * Hout * Wout
    dx_buf: []f32,  // N * in_c * H * W

    fn wIndex(self: *const ConvolutionalLayer, oc: usize, ic: usize, ky: usize, kx: usize) usize {
        // W[oc, ic, ky, kx] in (out_c, in_c, k_h, k_w)
        return (((oc * self.in_c + ic) * self.k_h + ky) * self.k_w + kx);
    }

    fn t4Index(n: usize, c: usize, y: usize, x: usize, C: usize, H: usize, W: usize) usize {
        // NCHW
        return (((n * C + c) * H + y) * W + x);
    }

    fn outDims(self: *const ConvolutionalLayer, hin: usize, win: usize) struct { h: usize, w: usize } {
        // floor((Hin + 2P - Kh)/S) + 1
        const h = (hin + 2 * self.pad - self.k_h) / self.stride + 1;
        const w = (win + 2 * self.pad - self.k_w) / self.stride + 1;
        return .{ .h = h, .w = w };
    }

    pub fn init(
        allocator: std.mem.Allocator,
        in_c: usize,
        out_c: usize,
        k_h: usize,
        k_w: usize,
        stride: usize,
        pad: usize,
        rng: *std.Random,
    ) !ConvolutionalLayer {
        if (in_c == 0 or out_c == 0 or k_h == 0 or k_w == 0 or stride == 0) {
            return error.InvalidArgs;
        }

        const w_len = out_c * in_c * k_h * k_w;

        var w = try allocator.alloc(f32, w_len);
        errdefer allocator.free(w);

        var b = try allocator.alloc(f32, out_c);
        errdefer allocator.free(b);

        var dw = try allocator.alloc(f32, w_len);
        errdefer allocator.free(dw);

        var db = try allocator.alloc(f32, out_c);
        errdefer allocator.free(db);

        // He-ish scale (good default for ReLU nets).
        const fan_in = @as(f32, @floatFromInt(in_c * k_h * k_w));
        const scale = std.math.sqrt(2.0 / fan_in);

        // init weights small random in [-scale, +scale], biases = 0
        for (w) |*val| {
            const u = rng.float(f32);         // [0,1)
            const r = (2.0 * u - 1.0) * scale; // [-scale, +scale]
            val.* = r;
        }
        @memset(b, 0.0);

        // grads start at 0
        @memset(dw, 0.0);
        @memset(db, 0.0);

        return .{
            .allocator = allocator,
            .in_c = in_c,
            .out_c = out_c,
            .k_h = k_h,
            .k_w = k_w,
            .stride = stride,
            .pad = pad,
            .w = w,
            .b = b,
            .dw = dw,
            .db = db,
            .cached_input = &[_]f32{},
            .cached_in_shape = .{ .n = 0, .c = 0, .h = 0, .w = 0 },
            .out_buf = &[_]f32{},
            .dx_buf = &[_]f32{},
        };
    }

    pub fn deinit(self: *ConvolutionalLayer) void {
        if (self.w.len != 0) self.allocator.free(self.w);
        if (self.b.len != 0) self.allocator.free(self.b);
        if (self.dw.len != 0) self.allocator.free(self.dw);
        if (self.db.len != 0) self.allocator.free(self.db);

        if (self.cached_input.len != 0) self.allocator.free(self.cached_input);
        if (self.out_buf.len != 0) self.allocator.free(self.out_buf);
        if (self.dx_buf.len != 0) self.allocator.free(self.dx_buf);

        self.* = undefined;
    }

    pub fn zeroGrad(self: *ConvolutionalLayer) void {
        @memset(self.dw, 0.0);
        @memset(self.db, 0.0);
    }

    pub fn forward(self: *ConvolutionalLayer, x: Tensor4) !Tensor4 {
        if (x.c != self.in_c) return TensorError.ShapeMismatch;

        // cache input for backward (copy)
        const x_len = x.data.len;
        if (self.cached_input.len != x_len) {
            if (self.cached_input.len != 0) self.allocator.free(self.cached_input);
            self.cached_input = try self.allocator.alloc(f32, x_len);
        }
        @memcpy(self.cached_input, x.data);

        self.cached_in_shape = .{ .n = x.n, .c = x.c, .h = x.h, .w = x.w };

        const od = self.outDims(x.h, x.w);
        const out_len = x.n * self.out_c * od.h * od.w;

        // ensure output buffer
        if (self.out_buf.len != out_len) {
            if (self.out_buf.len != 0) self.allocator.free(self.out_buf);
            self.out_buf = try self.allocator.alloc(f32, out_len);
        }

        var y = try Tensor4.init(self.out_buf, x.n, self.out_c, od.h, od.w);

        // compute convolution
        for (0..x.n) |n| {
            for (0..self.out_c) |oc| {
                for (0..od.h) |oy| {
                    for (0..od.w) |ox| {
                        var acc: f32 = self.b[oc];

                        for (0..self.in_c) |ic| {
                            for (0..self.k_h) |ky| {
                                const iy_i32: i32 = @intCast(oy * self.stride + ky);
                                const iy = iy_i32 - @as(i32, @intCast(self.pad));
                                if (iy < 0 or iy >= @as(i32, @intCast(x.h))) continue;

                                for (0..self.k_w) |kx| {
                                    const ix_i32: i32 = @intCast(ox * self.stride + kx);
                                    const ix = ix_i32 - @as(i32, @intCast(self.pad));
                                    if (ix < 0 or ix >= @as(i32, @intCast(x.w))) continue;

                                    const in_idx = t4Index(
                                        n,
                                        ic,
                                        @intCast(iy),
                                        @intCast(ix),
                                        x.c,
                                        x.h,
                                        x.w,
                                    );
                                    const wt = self.w[self.wIndex(oc, ic, ky, kx)];
                                    acc += self.cached_input[in_idx] * wt;
                                }
                            }
                        }

                        const out_idx = t4Index(n, oc, oy, ox, self.out_c, od.h, od.w);
                        y.data[out_idx] = acc;
                    }
                }
            }
        }

        return y;
    }

    pub fn backward(self: *ConvolutionalLayer, dout: Tensor4) !Tensor4 {
        // Expect dout matches forward output shape
        const in_sh = self.cached_in_shape;
        if (in_sh.n == 0) return error.MissingForwardCache;

        const od = self.outDims(in_sh.h, in_sh.w);

        if (dout.n != in_sh.n or dout.c != self.out_c or dout.h != od.h or dout.w != od.w) {
            return TensorError.ShapeMismatch;
        }

        // ensure dx buffer matches cached input size
        const dx_len = in_sh.n * in_sh.c * in_sh.h * in_sh.w;
        if (self.dx_buf.len != dx_len) {
            if (self.dx_buf.len != 0) self.allocator.free(self.dx_buf);
            self.dx_buf = try self.allocator.alloc(f32, dx_len);
        }
        @memset(self.dx_buf, 0.0);

        // Accumulate grads into dw/db (so call zeroGrad() before backward each step)
        for (0..in_sh.n) |n| {
            for (0..self.out_c) |oc| {
                for (0..od.h) |oy| {
                    for (0..od.w) |ox| {
                        const out_idx = t4Index(n, oc, oy, ox, self.out_c, od.h, od.w);
                        const g: f32 = dout.data[out_idx];

                        self.db[oc] += g;

                        for (0..self.in_c) |ic| {
                            for (0..self.k_h) |ky| {
                                const iy_i32: i32 = @intCast(oy * self.stride + ky);
                                const iy = iy_i32 - @as(i32, @intCast(self.pad));
                                if (iy < 0 or iy >= @as(i32, @intCast(in_sh.h))) continue;

                                for (0..self.k_w) |kx| {
                                    const ix_i32: i32 = @intCast(ox * self.stride + kx);
                                    const ix = ix_i32 - @as(i32, @intCast(self.pad));
                                    if (ix < 0 or ix >= @as(i32, @intCast(in_sh.w))) continue;

                                    const in_idx = t4Index(
                                        n,
                                        ic,
                                        @intCast(iy),
                                        @intCast(ix),
                                        in_sh.c,
                                        in_sh.h,
                                        in_sh.w,
                                    );

                                    const w_idx = self.wIndex(oc, ic, ky, kx);

                                    // dW += x * dout
                                    self.dw[w_idx] += self.cached_input[in_idx] * g;

                                    // dX += W * dout
                                    self.dx_buf[in_idx] += self.w[w_idx] * g;
                                }
                            }
                        }
                    }
                }
            }
        }

        return Tensor4.init(self.dx_buf, in_sh.n, in_sh.c, in_sh.h, in_sh.w);
    }
};
