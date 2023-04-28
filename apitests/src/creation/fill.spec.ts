import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import * as loader from "../load-tf";
import type tfTypes from "@tensorflow/tfjs-core";

describe("tf.fill(shape, value, dtype?): ", async () => {
  const tf: loader.TFModule = await loader.load();
  it("  -- basic", () => {
    const t: tfTypes.Tensor3D = tf.fill([3, 2, 3], 1);
    const expected = [
      [
        [1, 1, 1],
        [1, 1, 1],
      ],
      [
        [1, 1, 1],
        [1, 1, 1],
      ],
      [
        [1, 1, 1],
        [1, 1, 1],
      ],
    ];
    expect(t.arraySync()).to.eql(expected);
  });
  it("  -- dtype", () => {
    const t: tfTypes.Tensor2D = tf.fill([2, 1], 4, "int32");
    const expected = [[4], [4]];
    expect(t).to.haveDtype("int32");
    expect(t.arraySync()).to.eql(expected);
  });
});
