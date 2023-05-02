import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import * as loader from "../load-tf";
import type tfTypes from "@tensorflow/tfjs-core";

describe("tf.eye(numRows, numColumns?, batchShape?, dtype?): ", () => {
  it("  -- basic", async () => {
    const tf: loader.TFModule = await loader.load();
    const t: tfTypes.Tensor2D = tf.eye(3);
    const expected = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    expect(t.arraySync()).to.eql(expected);
  });
  it("  -- numColumns", async () => {
    const tf: loader.TFModule = await loader.load();
    const t: tfTypes.Tensor2D = tf.eye(3, 2);
    const expected = [
      [1, 0],
      [0, 1],
      [0, 0],
    ];
    expect(t.arraySync()).to.eql(expected);
  });
  it("  -- batchShape", async () => {
    const tf: loader.TFModule = await loader.load();
    const t: tfTypes.Tensor2D = tf.eye(3, undefined, [1, 1]);
    const identityMatrix = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const expected = [[identityMatrix]];
    expect(t.arraySync()).to.eql(expected);
  });
  it("  -- dtypes", async () => {
    const tf: loader.TFModule = await loader.load();
    const t: tfTypes.Tensor2D = tf.eye(3, undefined, undefined, "int32");
    const expected = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    expect(t).to.haveDtype("int32");
    expect(t.arraySync()).to.eql(expected);
  });
});
