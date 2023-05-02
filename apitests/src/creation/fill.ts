import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../load-tf";

let tf: loader.TFModule;

export function run() {
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  it("  -- basic", async () => {
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
  it("  -- dtype", async () => {
    const t: tfTypes.Tensor2D = tf.fill([2, 1], 4, "int32");
    const expected = [[4], [4]];
    expect(t).to.haveDtype("int32");
    expect(t.arraySync()).to.eql(expected);
  });
}
