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

/* -- tf.randomGamma (shape, alpha, beta?, dtype?, seed?) */
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

  it("  -- basic example: 2 alphas", () => {
    const shape = [2, 3];
    const lowAlpha = 1;
    const highAlpha = 8;
    // assertions:
    const lowTensorMeans: number[] = [];
    const highTensorMeans: number[] = [];
    for (let i = 0; i < NUM_EXAMPLES; i++) {
      const x: tfTypes.Tensor = tf.randomGamma(shape, lowAlpha);
      const lowMean = tf.mean(x).dataSync()[0];
      lowTensorMeans.push(lowMean);
      const y: tfTypes.Tensor = tf.randomGamma(shape, highAlpha);
      const highMean = tf.mean(y).dataSync()[0];
      highTensorMeans.push(highMean);
    }
    const netLowMean = average(lowTensorMeans);
    const netHighMean = average(highTensorMeans);
    expect(netLowMean).to.be.lessThan(netHighMean);
  });
}
