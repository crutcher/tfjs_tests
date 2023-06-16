import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.multinomial (logits, numSamples, seed?, normalized?) */
export function run() {
  /* HOOKS: */
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });

  /* TESTS: */

  it("  -- basic example", () => {
    const probs: tfTypes.Tensor1D = tf.tensor([0.75, 0.25]);
    const numSamples = 3;
    const expectedShape = [numSamples];
    // assertions:
    const result: tfTypes.Tensor = tf.multinomial(probs, numSamples);
    expect(result).to.haveShape(expectedShape);
  });

  it("  -- basic example: logits aren't normalized", () => {
    const probs: tfTypes.Tensor1D = tf.tensor([0.6, 0.25]);
    const numSamples = 3;
    const expectedShape = [numSamples];
    // assertions:
    const result: tfTypes.Tensor = tf.multinomial(probs, numSamples);
    expect(result).to.haveShape(expectedShape);
  });

  it("  -- basic example: 3 logits", () => {
    const probs: tfTypes.Tensor1D = tf.tensor([0.25, 0.25, 0.5]);
    const numSamples = 3;
    const expectedRange = [0, 1, 2];
    // assertions:
    const result: tfTypes.Tensor = tf.multinomial(probs, numSamples);
    expect(result).to.have.onlyValuesInSet(expectedRange);
  });

  it("  -- 2d logits example", () => {
    const probs: tfTypes.Tensor2D = tf.tensor([0.75, 0.25], [1, 2]);
    const numSamples = 3;
    const expectedShape = [1, numSamples];
    const expectedRange = [0, 1];
    // assertions:
    const result: tfTypes.Tensor = tf.multinomial(probs, numSamples);
    expect(result).to.haveShape(expectedShape);
    expect(result).to.have.onlyValuesInSet(expectedRange);
  });

  it("  -- 2d logits example: batch size 2", () => {
    const probs: tfTypes.Tensor2D = tf.tensor([0.1, 0.2, 0.4, 0.3], [2, 2]);
    const numSamples = 3;
    const expectedShape = [2, numSamples];
    const expectedRange = [0, 1];
    // assertions:
    const result: tfTypes.Tensor = tf.multinomial(probs, numSamples);
    expect(result).to.haveShape(expectedShape);
    expect(result).to.have.onlyValuesInSet(expectedRange);
  });

  it("  -- 2d logits example: batch size 4", () => {
    const probs: tfTypes.Tensor2D = tf.tensor([0.1, 0.2, 0.4, 0.3], [4, 1]);
    const numSamples = 3;
    const expectedShape = [4, numSamples];
    // assertions:
    const result: tfTypes.Tensor = tf.multinomial(probs, numSamples);
    expect(result).to.haveShape(expectedShape);
    expect(result).to.be.allZeros;
  });
  it("  -- 2d logits example: num outcomes 4", () => {
    const probs: tfTypes.Tensor2D = tf.tensor([0.1, 0.2, 0.4, 0.3], [1, 4]);
    const numSamples = 3;
    const expectedShape = [1, numSamples];
    const expectedRange = [0, 1, 2, 3];
    // assertions:
    const result: tfTypes.Tensor = tf.multinomial(probs, numSamples);
    expect(result).to.haveShape(expectedShape);
    expect(result).to.have.onlyValuesInSet(expectedRange);
  });

  it("  -- 2d logits example: normalized", () => {
    const probs: tfTypes.Tensor2D = tf.tensor([0.75, 0.25], [1, 2]);
    const numSamples = 3;
    const expectedShape = [1, numSamples];
    const expectedRange = [0, 1];
    // assertions:
    const result: tfTypes.Tensor = tf.multinomial(
      probs,
      numSamples,
      undefined,
      true
    );
    expect(result).to.haveShape(expectedShape);
    expect(result).to.have.onlyValuesInSet(expectedRange);
  });
}
