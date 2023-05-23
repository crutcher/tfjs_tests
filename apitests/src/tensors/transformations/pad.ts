import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* See also: src/tensors/transformations/mirrorPad.ts
  for more thorough testing of padding
*/

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.pad (x, paddings, constantValue?)-- */
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

  /* TESTS: */

  it("  -- default: pads with 0s", () => {
    const input = [1, 2, 3, 4];
    const padding: [number, number][] = [[1, 2]];
    const output = [0, 1, 2, 3, 4, 0, 0];
    const x: tfTypes.Tensor1D = tf.tensor1d(input);
    const y: tfTypes.Tensor1D = x.pad(padding);
    expect(y).to.lookLike(output);
  });
  it("  -- constantValue: pads with constantValue", () => {
    const constantValue = 9;
    const input = [1, 2, 3, 4];
    const padding: [number, number][] = [[1, 2]];
    const output = [9, 1, 2, 3, 4, 9, 9];
    const x: tfTypes.Tensor1D = tf.tensor1d(input);
    const y: tfTypes.Tensor1D = x.pad(padding, constantValue);
    expect(y).to.lookLike(output);
  });
}
