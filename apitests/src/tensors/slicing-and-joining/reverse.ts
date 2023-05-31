import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.reverse (x, axis?)) */
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
    const initialValues = [1, 2, 3, 4];
    const expectedResult = [...initialValues].reverse();
    const x: tfTypes.Tensor1D = tf.tensor1d(initialValues);
    // assertions:
    const result: tfTypes.Tensor1D = x.reverse();
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- 1d example: rank specific methods", async () => {
    const initialValues = [1, 2, 3, 4];
    const expectedResult = [...initialValues].reverse();
    const x: tfTypes.Tensor1D = tf.tensor1d(initialValues);
    // assertions:
    const result: tfTypes.Tensor1D = tf.reverse1d(x);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- 2d example: basic", async () => {
    // reverses all dimensions recursively
    const initialValues = [
      [1, 2],
      [3, 4],
    ];
    const expectedResult = [
      [4, 3],
      [2, 1],
    ];
    const x: tfTypes.Tensor2D = tf.tensor2d(initialValues);
    // assertions:
    const result: tfTypes.Tensor2D = x.reverse();
    expect(result).to.lookLike(expectedResult);
  });
  it("  -- 2d example: axis: reverse only 0th dimension", async () => {
    const axis = 0;
    const initialValues = [
      [1, 2],
      [3, 4],
    ];
    const expectedResult = [
      [3, 4],
      [1, 2],
    ];
    const x: tfTypes.Tensor2D = tf.tensor2d(initialValues);
    // assertions:
    const result: tfTypes.Tensor2D = x.reverse(axis);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- 2d example: axis: reverse only 1st dimension", async () => {
    const axis = 1;
    const initialValues = [
      [1, 2],
      [3, 4],
    ];
    const expectedResult = [
      [2, 1],
      [4, 3],
    ];
    const x: tfTypes.Tensor2D = tf.tensor2d(initialValues);
    // assertions:
    const result: tfTypes.Tensor2D = x.reverse(axis);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- 2d example: axis: ERROR - bad dimension", async () => {
    const outOfRange = 4;
    const axis = outOfRange;
    const initialValues = [
      [1, 2],
      [3, 4],
    ];
    const x: tfTypes.Tensor2D = tf.tensor2d(initialValues);
    // assertions:
    expect(() => x.reverse(axis)).to.throw(
      `All values in axis param must be in range [-2, 2) but got axis 4`
    );
  });
}
