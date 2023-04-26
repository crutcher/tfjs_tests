import { expect } from "chai";
import * as tf from "@tensorflow/tfjs";

describe("tf.complex(): ", () => {
  it("  -- converts 2 real numbers to a complex number", () => {
    const real: tf.Tensor2D = tf.tensor2d([
      [0, 1],
      [2, 3],
    ]);
    const imag: tf.Tensor2D = tf.tensor2d([
      [0, 10],
      [20, 30],
    ]);
    const complex: tf.Tensor2D = tf.complex(real, imag);
    expect(complex.dtype).to.equal("complex64");
    expect(complex.size).to.equal(4);
  });
});
