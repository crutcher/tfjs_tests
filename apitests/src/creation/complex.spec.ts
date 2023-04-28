import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
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
    expect(complex).to.haveDtype("complex64");
    expect(complex.size).to.equal(4);
  });
});
