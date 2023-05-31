import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.concat (tensors, axis?) */
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

  it("  -- 1d example: basic", async () => {
    const a: tfTypes.Tensor1D = tf.tensor1d([1, 2]);
    const b: tfTypes.Tensor1D = tf.tensor1d([3, 4]);
    const expectedValues = [1, 2, 3, 4];
    const expectedShape = [4];
    // assertions:
    const c: tfTypes.Tensor1D = a.concat(b);
    expect(c).to.lookLike(expectedValues);
    expect(c).to.haveShape(expectedShape);
  });
  it("  -- 1d example: concat 3 tensors", async () => {
    const a: tfTypes.Tensor1D = tf.tensor1d([1, 2]);
    const b: tfTypes.Tensor1D = tf.tensor1d([3, 4]);
    const c: tfTypes.Tensor1D = tf.tensor1d([5, 6]);
    const expectedValues = [1, 2, 3, 4, 5, 6];
    const expectedShape = [6];
    // assertions:
    const d: tfTypes.Tensor1D = tf.concat([a, b, c]);
    expect(d).to.lookLike(expectedValues);
    expect(d).to.haveShape(expectedShape);
  });
  it("  -- 1d example: rank specific", async () => {
    const a: tfTypes.Tensor1D = tf.tensor1d([1, 2]);
    const b: tfTypes.Tensor1D = tf.tensor1d([3, 4]);
    const expectedValues = [1, 2, 3, 4];
    const expectedShape = [4];
    // assertions:
    const c: tfTypes.Tensor1D = tf.concat([a, b]);
    expect(c).to.lookLike(expectedValues);
    expect(c).to.haveShape(expectedShape);
  });
  it("  -- 1d example: different sized tensors", async () => {
    const a: tfTypes.Tensor1D = tf.tensor1d([1, 2]);
    const b: tfTypes.Tensor1D = tf.tensor1d([3, 4, 5, 6]);
    const expectedValues = [1, 2, 3, 4, 5, 6];
    const expectedShape = [6];
    // assertions:
    const c: tfTypes.Tensor1D = a.concat(b);
    expect(c).to.lookLike(expectedValues);
    expect(c).to.haveShape(expectedShape);
  });
  it("  -- 2d example: basic", async () => {
    const a: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    const b: tfTypes.Tensor2D = tf.tensor2d([
      [5, 6],
      [7, 8],
    ]);
    const expectedValues = [
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ];
    const expectedShape = [4, 2];
    // assertions:
    const c: tfTypes.Tensor2D = a.concat(b);
    expect(c).to.lookLike(expectedValues);
    expect(c).to.haveShape(expectedShape);
  });
  it("  -- 2d example: with axis", async () => {
    const axis = 1;
    const a: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    const b: tfTypes.Tensor2D = tf.tensor2d([
      [5, 6],
      [7, 8],
    ]);
    const expectedValues = [
      [1, 2, 5, 6],
      [3, 4, 7, 8],
    ];
    const expectedShape = [2, 4];
    // assertions:
    const c: tfTypes.Tensor2D = a.concat(b, axis);
    expect(c).to.lookLike(expectedValues);
    expect(c).to.haveShape(expectedShape);
  });
  it("  -- error: mismatched ranks", async () => {
    const a: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    const b: tfTypes.Tensor1D = tf.tensor1d([5, 6]);
    // assertions:
    expect(() => a.concat(b)).to.throw(
      `Error in concat2D: rank of tensors[1] must be the same as the rank of the rest (2)`
    );
  });
}
