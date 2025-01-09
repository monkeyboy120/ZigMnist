//NOTE: Tensor Impementation

const std = @import("std");

pub const Tensor = struct {
    data: []f64,
    shape: [3]usize, // Height x Width x Channels

    pub fn new(data: []f64, shape: [3]usize) Tensor {
        return Tensor{
            .data = data,
            .shape = shape,
        };
    }

    pub fn at(self: Tensor, h: usize, w: usize, c: usize) !f64 {
        if (h >= self.shape[0] or w >= self.shape[1] or c >= self.shape[2]) {
            return error.IndexOutOfBounds;
        }
        const index = h * self.shape[1] * self.shape[2] + w * self.shape[2] + c;
        return self.data[index];
    }

    pub fn set(self: Tensor, h: usize, w: usize, c: usize, value: f64) !void {
        if (h >= self.shape[0] or w >= self.shape[1] or c >= self.shape[2]) {
            return error.IndexOutOfBounds;
        }
        const index = h * self.shape[1] * self.shape[2] + w * self.shape[2] + c;
        self.data[index] = value;
    }
};
