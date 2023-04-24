import { expect } from "chai";
import * as tf from '@tensorflow/tfjs';

// See: https://js.tensorflow.org/api/latest/#tensor
describe("tf.tensorNd() : n∈{1, 2, 3, 4, 5, 6}", () => {
  it("  -- bad values", () => {
    expect(() => tf.tensor1d([[1], [2]] as any)).to.throw(
      "requires values to be a flat/TypedArray"
    );
  });
  it("  -- tf.tensor1d()", () => {
    const t: tf.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    expect(t.dtype).to.equal("float32");
    expect(t.shape).to.eql([4]);
    expect(t.arraySync()).to.eql([1, 2, 3, 4]);
  });
  it("  -- tf.tensor2d()", () => {
    const t: tf.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    expect(t.dtype).to.equal("float32");
    expect(t.shape).to.eql([2,2]);
    expect(t.arraySync()).to.eql([
      [1, 2],
      [3, 4],
    ]);
  });
});
