import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import * as loader from "../load-tf";
import type tfTypes from "@tensorflow/tfjs-core";

/* ---- tf.linspace(start, stop, num) ---- *
  Return an evenly spaced sequence of numbers over the given interval.
*/
describe("tf.linspace(start, stop, num): ", () => {
  // CONSTANTS
  const START = 0;
  const STOP = 0;
  const NUM = 10;

  // TESTS
  it("  -- basic", async () => {
    const tf: loader.TFModule = await loader.load();
    const t: tfTypes.Tensor1D = tf.linspace(START, STOP, NUM);
    t.print();
  });
});
