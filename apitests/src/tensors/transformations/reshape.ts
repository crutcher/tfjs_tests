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

/* -- tf.reshape (x, shape)-- */
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

  it("  -- default", () => {
    const input = [1, 2, 3, 4];
    const newShape = [2, 2];
    const output = [
      [1, 2],
      [3, 4],
    ];
    const x: tfTypes.Tensor1D = tf.tensor1d(input);
    const y: tfTypes.Tensor2D = x.reshape(newShape);
    expect(y).to.lookLike(output);
  });
  it("  -- with dimension set to -1", () => {
    //input shape = [ 3, 2, 2 ]
    const input = [
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
    const newShape = [4, -1];
    // -1 should be inferred to 12/4 = 3
    const expectedShape = [4, 3];
    const output = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ];
    const x: tfTypes.Tensor3D = tf.tensor3d(input);
    const y: tfTypes.Tensor2D = x.reshape(newShape);
    expect(y).to.haveShape(expectedShape);
    expect(y).to.lookLike(output);
  });
  it("  -- more than 1 dimension set to -1 throws error", () => {
    const newShape = [2, -1, -1];
    // (use same input as previous test)
    const x: tfTypes.Tensor3D = tf.range(1, 13).reshape([3, 2, 2]);
    expect(() => x.reshape(newShape)).to.throw(
      `Shapes can only have 1 implicit size. Found -1 at dim 1 and dim 2`
    );
  });
  it("  -- a shape of [-1] flattens array", () => {
    const newShape = [-1];
    const output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    // (use same input as previous test)
    const x: tfTypes.Tensor3D = tf.range(1, 13).reshape([3, 2, 2]);
    const y: tfTypes.Tensor1D = x.reshape(newShape);
    expect(y).to.lookLike(output);
  });
}
