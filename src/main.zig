const std = @import("std");
const MNIST = @import("mnist.zig");
const NN = @import("nn.zig").NN;
const SGD = @import("optimizer.zig").SGD;
const Loss = @import("loss.zig").Loss;
const Tensor = @import("tensor.zig").Tensor;
const FCLayer = @import("fc.zig").FCLayer;

pub fn main() !void {
    var allocator = std.heap.page_allocator;

    // Load the MNIST dataset
    const mnist = try MNIST.readMnistData(&allocator);
    defer mnist.destruct(&allocator);

    // Define the neural network architecture
    const fc1 = try FCLayer.new(784, 128, &allocator); // Input: 784, Output: 128
    const fc2 = try FCLayer.new(128, 10, &allocator); // Input: 128, Output: 10
    const network = NN.new(&[_]FCLayer{ fc1, fc2 });

    // Initialize the optimizer
    const optimizer = SGD.new(0.01); // Learning rate = 0.01

    // Training parameters
    const epochs = 5;
    const batch_size = 64;

    // Training loop
    for (epochs) |epoch| {
        std.debug.print("Epoch {d}/{d}\n", .{ epoch + 1, epochs });

        var batch_start: usize = 0;
        while (batch_start < mnist.trainImages.len / 784) {
            const batch_end = std.math.min(batch_start + batch_size, mnist.trainImages.len / 784);

            // Extract input and label batch
            const input = Tensor.new(mnist.trainImages.slice(batch_start * 784, batch_end * 784), .{ batch_end - batch_start, 784 });
            const labels = Tensor.new(mnist.trainLabels.slice(batch_start, batch_end), .{ batch_end - batch_start, 10 });

            // Forward pass
            const predictions = try network.forward(input, &allocator);

            // Compute loss
            const loss = try Loss.crossEntropy(labels.data, predictions.data);
            std.debug.print("Loss: {f}\n", .{loss});

            // Backpropagation and weight update (assuming `network.backward` exists)
            const gradients = try network.backward(labels, &allocator);
            try optimizer.step(network.getParams(), gradients);

            // Clean up
            defer allocator.free(predictions.data);
            defer allocator.free(gradients);

            batch_start += batch_size;
        }
    }

    std.debug.print("Training complete!\n", .{});
}
