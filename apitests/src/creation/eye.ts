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
    const t: tfTypes.Tensor2D = tf.eye(3);
    const expected = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    expect(t.arraySync()).to.eql(expected);
  });
  it("  -- numColumns", async () => {
    const t: tfTypes.Tensor2D = tf.eye(3, 2);
    const expected = [
      [1, 0],
      [0, 1],
      [0, 0],
    ];
    expect(t.arraySync()).to.eql(expected);
  });
  it("  -- batchShape", async () => {
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
    const t: tfTypes.Tensor2D = tf.eye(3, undefined, undefined, "int32");
    const expected = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    expect(t).to.haveDtype("int32");
    expect(t.arraySync()).to.eql(expected);
  });
}
