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
  // CONSTANTS
  const START = 0;
  const STOP = 9;
  const NUMS_RESULTS: { [key: string]: number[] } = {
    4: [0, 3, 6, 9],
    10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  };

  // TESTS
  it("  -- basic", async () => {
    for (const key in NUMS_RESULTS) {
      const expected = NUMS_RESULTS[key];
      const num = Number(key);
      const t: tfTypes.Tensor = tf.linspace(START, STOP, num);
      expect(t).to.haveShape([num]);
      expect(t).to.lookLike(expected);
    }
  });
}
