import { expect } from "chai";
import * as tf from "@tensorflow/tfjs";

// See: https://js.tensorflow.org/api/latest/#tensor
describe("tf.tensor(): ", () => {
  it("  -- default options", () => {
    const t: tf.Tensor<tf.Rank.R1> = tf.tensor([2, 3]);
    expect(t.dtype).to.equal("float32");
  });
  it("  -- shapes", () => {
    const t: tf.Tensor<tf.Rank.R2> = tf.tensor([2, 3, 4, 5], [2, 2]);
    expect(t.dtype).to.equal("float32");
    expect(t.shape).to.eql([2, 2]);
    expect(t.arraySync()).to.eql([
      [2.0, 3.0],
      [4.0, 5.0],
    ]);
  });
  it("  -- dtypes", () => {
    const t: tf.Tensor<tf.Rank.R2> = tf.tensor(
      [
        [2, 3],
        [4, 5],
      ],
      undefined,
      "int32"
    );
    expect(t.dtype).to.equal("int32");
    expect(t.shape).to.eql([2, 2]);
    expect(t.arraySync()).to.eql([
      [2, 3],
      [4, 5],
    ]);
  });
});
