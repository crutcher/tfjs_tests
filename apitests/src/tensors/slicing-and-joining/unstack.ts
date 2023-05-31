import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.unstack (x, axis?) */
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

  it("  -- 2d example: basic", async () => {
    const a: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const expectedLength = 2;
    const expectedShape = [4];
    // assertions:
    const results: tfTypes.Tensor<tfTypes.Rank>[] = tf.unstack(a);
    expect(results).to.have.lengthOf(expectedLength);
    results.forEach((result) => {
      expect(result).to.haveShape(expectedShape);
    });
  });
  it("  -- 2d example: with axis", async () => {
    const axis = 1;
    const a: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const expectedLength = 4;
    const expectedShape = [2];
    // assertions:
    const results: tfTypes.Tensor<tfTypes.Rank>[] = tf.unstack(a, axis);
    expect(results).to.have.lengthOf(expectedLength);
    results.forEach((result) => {
      expect(result).to.haveShape(expectedShape);
    });
  });
  it("  -- 2d example: negative axis", async () => {
    const axis = -2;
    // same result as 0. axis = -1 is the same as axis = 1
    const a: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const expectedLength = 2;
    const expectedShape = [4];
    // assertions:
    const results: tfTypes.Tensor<tfTypes.Rank>[] = tf.unstack(a, axis);
    expect(results).to.have.lengthOf(expectedLength);
    results.forEach((result) => {
      result.print();
      expect(result).to.haveShape(expectedShape);
    });
  });
  it("  -- !! 2d example: bad axis", async () => {
    const axis = 5;
    const a: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    // assertions:
    expect(() => tf.unstack(a, axis)).to.throw(`Axis = 5 is not in [-2, 2)`);
  });
}
