import { expect } from "chai";
import loadTf from "../load-tf";
import type tfTypes from "@tensorflow/tfjs";

describe("tf.eye(numRows, numColumns?, batchShape?, dtype?): ", async () => {
  const tf = await loadTf();
  it("  -- basic", () => {
    const t: tfTypes.Tensor2D = tf.eye(3);
    const expected = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    expect(t.arraySync()).to.eql(expected);
  });
});
