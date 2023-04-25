import { expect } from "chai";
import * as tf from "@tensorflow/tfjs";

describe("tf.clone(): ", () => {
  it("  -- clones a tensor", () => {
    const arr = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const t: tf.Tensor = tf.tensor(arr);
    const clone: tf.Tensor = t.clone();
    expect(clone.shape).to.eql([2, 3]);
  });
});
