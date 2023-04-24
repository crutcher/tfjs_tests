import { expect } from "chai";
import * as tf from "@tensorflow/tfjs";

// See: https://js.tensorflow.org/api/latest/#tensor
describe("tf.scalar(): ", () => {
  it("  -- default options", () => {
    const t: tf.Scalar = tf.scalar(2);
    expect(t.dtype).to.equal("float32");
    expect(t.shape).to.eql([]);
    expect(t.arraySync()).to.eql(2);
  });
  it("  -- shapes", () => {
    const t: tf.Scalar = tf.scalar(true, "bool");
    expect(t.dtype).to.equal("bool");
    expect(t.shape).to.eql([]);
    // Note: arraySync converts bools to numbers (0 or 1)
    expect(t.arraySync()).to.eql(1);
  });
});
