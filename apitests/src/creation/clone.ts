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
  it("  -- clones a tensor", () => {
    const arr = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const t: tfTypes.Tensor = tf.tensor(arr);
    const clone: tfTypes.Tensor = t.clone();
    expect(clone).to.haveShape([2, 3]);
  });
}
