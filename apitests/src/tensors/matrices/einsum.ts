import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.einsum (equation, ...tensors) */
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

  it("  -- matrix multiplication", () => {
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const y: tfTypes.Tensor2D = tf.tensor2d([
      [0, 1],
      [2, 3],
      [4, 5],
    ]);
    const expected = [
      [16, 22],
      [34, 49],
    ];
    // assertions:
    const result: tfTypes.Tensor = tf.einsum("ij,jk->ik", x, y);
    expect(result).to.lookLike(expected);
  });
  it("  -- !! matrix multiplication: bad matrix shapes", () => {
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const y: tfTypes.Tensor2D = tf.tensor2d([
      [0, 1],
      [2, 3],
      [4, 5],
      [6, 7],
    ]);
    // assertions:
    expect(() => tf.einsum("ij,jk->ik", x, y)).to.throw(
      `Expected dimension 3 at axis 0 of input shaped [4,2], but got dimension 4`
    );
  });
  it("  -- dot product (scalar result)", () => {
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3]);
    const y: tfTypes.Tensor1D = tf.tensor1d([0, 1, 2]);
    const expected = 8;
    const expectedShape: number[] = [];
    // assertions:
    const result: tfTypes.Tensor = tf.einsum("i,i->", x, y);
    expect(result).to.lookLike(expected);
    expect(result).to.haveShape(expectedShape);
  });
}
