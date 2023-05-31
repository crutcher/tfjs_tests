import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.slice (x, begin, size?) */
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
    const begin = [1];
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    const expected = [2, 3, 4];
    // assertions:
    const result: tfTypes.Tensor1D = x.slice(begin);
    expect(result).to.lookLike(expected);
  });

  it("  -- size: basic 1d example", async () => {
    const size = [2];
    const begin = [1];
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    const expected = [2, 3];
    // assertions:
    const result: tfTypes.Tensor1D = x.slice(begin, size);
    expect(result).to.lookLike(expected);
  });

  it("  -- begin out of range: tensor with negative size and shape", async () => {
    const begin = [42];
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    const expectedShape = [-38];
    const expectedSize = -38;
    // assertions:
    const result: tfTypes.Tensor1D = x.slice(begin);
    expect(result).to.haveShape(expectedShape);
    expect(result).to.haveSize(expectedSize);
  });

  it("  -- begin out of range: string representation is empty array", async () => {
    const begin = [42];
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    // assertions:
    const result: tfTypes.Tensor1D = x.slice(begin);
    expect(result.toString()).to.eql("Tensor\n    []");
  });

  it("  -- begin out of range: string representation is empty array", async () => {
    const begin = [42];
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    // assertions:
    const result: tfTypes.Tensor1D = x.slice(begin);
    expect(result.toString()).to.eql("Tensor\n    []");
  });

  it("  -- !! begin out of range: arraySync() / array() throw error", async () => {
    const begin = [42];
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    // assertions:
    const result: tfTypes.Tensor1D = x.slice(begin);
    expect(() => result.arraySync()).to.throw(
      `[-38] does not match the input size 0.`
    );
  });

  it("  -- 1d example: begin as single number", async () => {
    // equivalent to [1]
    const begin = 1;
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    const expected = [2, 3, 4];
    // assertions:
    const result: tfTypes.Tensor1D = x.slice(begin);
    expect(result).to.lookLike(expected);
  });

  it("  -- 2d example: basic", async () => {
    const begin = [1];
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const expected = [
      [3, 4],
      [5, 6],
      [7, 8],
    ];
    // assertions:
    const result: tfTypes.Tensor2D = x.slice(begin);
    expect(result).to.lookLike(expected);
  });

  it("  -- 2d example: 2d begin value", async () => {
    const begin = [0, 1];
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const expected = [[2], [4], [6], [8]];
    // assertions:
    const result: tfTypes.Tensor2D = x.slice(begin);
    expect(result).to.lookLike(expected);
  });

  it("  -- size: basic 2d example", async () => {
    const size = [2];
    const begin = [1];
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const expected = [
      [3, 4],
      [5, 6],
    ];
    // assertions:
    const result: tfTypes.Tensor2D = x.slice(begin, size);
    expect(result).to.lookLike(expected);
  });

  it("  -- size: 2d size value", async () => {
    const size = [2, 1];
    const begin = [1];
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const expected = [[3], [5]];
    // assertions:
    const result: tfTypes.Tensor2D = x.slice(begin, size);
    expect(result).to.lookLike(expected);
  });
}
