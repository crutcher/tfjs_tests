import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.spaceToBatchND (x, blockShape, paddings)-- */
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

  it("  -- default", async () => {
    const x: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const blockShape = [2, 2];
    const paddings = [
      [0, 0],
      [0, 0],
    ];
    const expectedResult = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]];
    const expectedShape = [4, 1, 1, 1];
    const y = x.spaceToBatchND(blockShape, paddings);
    expect(y).to.haveShape(expectedShape);
    expect(y).to.lookLike(expectedResult);
  });

  it("  -- paddings : 1st dimension", async () => {
    // looks like paddings are added to original tensor before being mutated
    const x: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const blockShape = [2, 2];
    const paddings = [
      [1, 1],
      [0, 0],
    ];
    const expectedResult = [
      [[[0]], [[3]]],

      [[[0]], [[4]]],

      [[[1]], [[0]]],

      [[[2]], [[0]]],
    ];
    const expectedShape = [4, 2, 1, 1];
    //assertions:
    const y = x.spaceToBatchND(blockShape, paddings);
    expect(y).to.haveShape(expectedShape);
    expect(y).to.lookLike(expectedResult);
  });

  it("  -- paddings : 1st dimension only left padding", async () => {
    const x: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const blockShape = [2, 2];
    const paddings = [
      [2, 0],
      [0, 0],
    ];
    const expectedResult = [
      [[[0]], [[1]]],

      [[[0]], [[2]]],

      [[[0]], [[3]]],

      [[[0]], [[4]]],
    ];
    const expectedShape = [4, 2, 1, 1];
    // assertions:
    const y = x.spaceToBatchND(blockShape, paddings);
    expect(y).to.haveShape(expectedShape);
    expect(y).to.lookLike(expectedResult);
  });

  it("  -- paddings : 2nd dimension", async () => {
    const x: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const blockShape = [2, 2];
    const paddings = [
      [0, 0],
      [2, 0],
    ];
    const expectedResult = [
      [[[0], [1]]],

      [[[0], [2]]],

      [[[0], [3]]],

      [[[0], [4]]],
    ];
    const expectedShape = [4, 1, 2, 1];
    // assertions:
    const y = x.spaceToBatchND(blockShape, paddings);
    expect(y).to.haveShape(expectedShape);
    expect(y).to.lookLike(expectedResult);
  });

  it("  -- bad padding: error", async () => {
    const x: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const blockShape = [2, 2];
    const badPaddings = [
      [1, 0],
      [0, 0],
    ];
    // assertions:
    expect(() => x.spaceToBatchND(blockShape, badPaddings)).to.throw(
      `input spatial dimensions 2,2,1 with paddings 1,0,0,0 must be divisible by blockShapes 2,2`
    );
  });
}
