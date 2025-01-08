// Implements loss functions

const std = @import("std");

pub const Loss = struct {
    pub fn meanSquaredError(y_true: []const f64, y_pred: []const f64) !f64 {
        if (y_true.len != y_pred.len) {
            return error.MismatchedLengths;
        }

        var sum: f64 = 0.0;
        for (y_true) |i| {
            const pred_val = y_pred[i];
            sum += std.math.pow(y_true[i] - pred_val, 2);
        }

        return sum / y_true.len;
    }
    pub fn crossEntropyLoss(y_true: []const f64, y_pred: []const f64) !f64 {
        if (y_true.len != y_pred.len) {
            return error.MismatchedLengths;
        }

        var loss: f64 = 0.0;

        for (y_true) |i| {
            const pred_val = y_pred[i];

            if (pred_val <= 0.0 or pred_val >= 1.0) {
                return error.InvalidProbability;
            }

            loss += -y_true[i] * @log(pred_val);
        }

        return loss;
    }
};
