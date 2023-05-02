import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import * as loader from "../load-tf";
import type tfTypes from "@tensorflow/tfjs-core";

/* ---- tf.linspace(start, stop, num)---- *
  Return an evenly spaced sequence of numbers (including decimals) over the given interval.
*/
describe("tf.linspace(start, stop, num): ", () => {
  // CONSTANTS
  const START = 0;
  const STOP = 9;
  const NUMS_RESULTS: { [key: string]: number[] } = {
    4: [0, 3, 6, 9],
    10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  };

  // TESTS
  it("  -- basic", async () => {
    const tf: loader.TFModule = await loader.load();

    for (const key in NUMS_RESULTS) {
      const expected = NUMS_RESULTS[key];
      const num = Number(key);
      const t: tfTypes.Tensor = tf.linspace(START, STOP, num);
      expect(t).to.haveShape([num]);
      expect(t).to.lookLike(expected);
    }
  });
});
