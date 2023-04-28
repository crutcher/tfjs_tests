import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import * as tf from "@tensorflow/tfjs";

describe("tf.clone(): ", () => {
  it("  -- clones a tensor", () => {
    const arr = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const t: tf.Tensor = tf.tensor(arr);
    const clone: tf.Tensor = t.clone();
    expect(clone).to.haveShape([2, 3]);
  });
});
