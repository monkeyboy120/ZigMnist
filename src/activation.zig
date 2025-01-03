// Activation Functions

const std = @import("std");

// Relu activation function
pub fn relu(value: f64) f64 {
    return if (value > 0) value else 0;
}

// Sigmoid activation function
pub fn sigmoid(value: f64) f64 {
    return 1.0 / (1.0 + std.math.exp(-value));
}

pub fn softmax(input: []const f64, allocator: *std.mem.Allocator) ![]f64 {
    // Find the maximum value for numerical stability
    var max_val = -std.math.inf(f64);
    for (input) |value| {
        if (value > max_val) {
            max_val = value;
        }
    }

    // Allocate memory for the output
    var output = try allocator.alloc(f64, input.len);

    // Compute the exponentials and their sum
    var exp_sum: f64 = 0.0;
    for (input) |value| {
        const exp_val = @exp(value - max_val);
        output[&value - &input[0]] = exp_val; // Store the value
        exp_sum += exp_val;
    }

    // Normalize to make it a probability distribution
    for (output) |*value| {
        value.* /= exp_sum;
    }

    return output;
}
