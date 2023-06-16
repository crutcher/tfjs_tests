import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

import { average } from "../../utils/general-utils";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.randomUniform (shape, minval?, maxval?, dtype?, seed?) */
export function run() {
  /* HOOKS: */
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });

  /* CONSTANTS: */
  const NUM_EXAMPLES = 20;

  /* TESTS: */

  it("  -- basic example", () => {
    const shape = [2, 3];
    const expectedRange = [0, 1];
    // assertions:
    const results: tfTypes.Tensor[] = [];
    for (let i = 0; i < NUM_EXAMPLES; i++) {
      const x: tfTypes.Tensor = tf.randomUniform(shape);
      results.push(x);
    }
    const means = results.map((t) => tf.mean(t).dataSync()[0]);
    const totalMean = average(means);
    expect(totalMean).to.be.within(expectedRange[0], expectedRange[1]);
  });

  it("  -- basic example: with min and max", () => {
    const shape = [2, 3];
    const minVal = 16;
    const maxVal = 32;
    const expectedRange = [minVal, maxVal];
    // assertions:
    const results: tfTypes.Tensor[] = [];
    for (let i = 0; i < NUM_EXAMPLES; i++) {
      const x: tfTypes.Tensor = tf.randomUniform(shape, minVal, maxVal);
      results.push(x);
    }
    const means = results.map((t) => tf.mean(t).dataSync()[0]);
    const totalMean = average(means);
    expect(totalMean).to.be.within(expectedRange[0], expectedRange[1]);
  });
}
