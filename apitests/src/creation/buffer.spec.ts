import { expect } from "chai";
import * as tf from "@tensorflow/tfjs";

/* tf.TensorBuffer: A mutable object, similar to tf.Tensor, that allows users to set values at locations before converting to an immutable tf.Tensor. */
describe("tf.buffer(): ", () => {
  it("  -- default options", () => {
    const buffer: tf.TensorBuffer<tf.Rank.R2> = tf.buffer([2, 2]);
    expect(buffer.shape).to.eql([2, 2]);
    expect(buffer.size).to.eql(4);
    buffer.set(4, 0, 0);
    buffer.set(6, 0, 1);
    buffer.set(8, 1, 0);
    buffer.set(10, 1, 1);
    const expected = [
      [4, 6],
      [8, 10],
    ];
    expect(buffer.toTensor().arraySync()).to.eql(expected);
  });
  it("  -- dtype", () => {
    const buffer: tf.TensorBuffer<tf.Rank.R2, "int32"> = tf.buffer(
      [2, 2],
      "int32"
    );
    expect(buffer.dtype).to.equal("int32");
  });
  it("  -- values", () => {
    const buffer: tf.TensorBuffer<tf.Rank.R2, keyof tf.DataTypeMap> = tf.buffer(
      [2, 2],
      undefined,
      Int32Array.from([1, 2, 3, 4])
    );
    const expected = [
      [1, 2],
      [3, 4],
    ];
    expect(buffer.toTensor().arraySync()).to.eql(expected);
  });
});
