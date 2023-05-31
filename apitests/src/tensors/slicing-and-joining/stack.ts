import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.stack (tensors, axis?) */
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
    const c: tfTypes.Tensor1D = tf.tensor1d([5, 6]);
    const expectedResult = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    // assertions:
    const result = tf.stack([a, b, c]);
    expect(result).to.haveShape([3, 2]);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- !! bad example: shapes don't match", async () => {
    const a: tfTypes.Tensor1D = tf.tensor1d([2]);
    const b: tfTypes.Tensor1D = tf.tensor1d([3, 4]);
    const c: tfTypes.Tensor1D = tf.tensor1d([5, 6]);
    // assertions:
    expect(() => tf.stack([a, b, c])).to.throw(
      `All tensors passed to stack must have matching shapes Shapes 1 and 2 must match`
    );
  });

  it("  -- 2d example: with axis", async () => {
    const a: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b: tfTypes.Tensor2D = tf.tensor2d([5, 6, 7, 8], [2, 2]);
    const c: tfTypes.Tensor2D = tf.tensor2d([9, 10, 11, 12], [2, 2]);
    const expectedResult = [
      [
        [1, 2],
        [3, 4],
      ],

      [
        [5, 6],
        [7, 8],
      ],

      [
        [9, 10],
        [11, 12],
      ],
    ];
    // assertions:
    const result = tf.stack([a, b, c]);
    expect(result).to.haveShape([3, 2, 2]);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- 2d example: with axis", async () => {
    const axis = 1;
    const a: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b: tfTypes.Tensor2D = tf.tensor2d([5, 6, 7, 8], [2, 2]);
    const c: tfTypes.Tensor2D = tf.tensor2d([9, 10, 11, 12], [2, 2]);
    const expectedResult = [
      [
        [1, 2],
        [5, 6],
        [9, 10],
      ],

      [
        [3, 4],
        [7, 8],
        [11, 12],
      ],
    ];
    // assertions:
    const result = tf.stack([a, b, c], axis);
    expect(result).to.haveShape([2, 3, 2]);
    expect(result).to.lookLike(expectedResult);
  });
}
