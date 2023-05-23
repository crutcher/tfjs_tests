import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import { default as chaiAsPromised } from "chai-as-promised";
chai.use(chaiAsPromised);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* See also: src/tensors/transformations/mirrorPad.ts
  for more thorough testing of padding
*/

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.setdiff1dAsync (x, y)-- */
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
    const x = [1, 2, 3, 4, 5, 6];
    const y = [1, 3, 5];
    const expectedOut = [2, 4, 6];
    const expectedIndices = [1, 3, 5];
    const [out, indices]: [tfTypes.Tensor, tfTypes.Tensor] =
      await tf.setdiff1dAsync(x, y);
    expect(out).to.lookLike(expectedOut);
    expect(indices).to.lookLike(expectedIndices);
  });
  it("  -- is not commutative", async () => {
    const x = [1, 3, 5];
    const y = [1, 2, 3, 4, 5, 6];
    const expectedOut: number[] = [];
    const expectedIndices: number[] = [];
    const [out, indices]: [tfTypes.Tensor, tfTypes.Tensor] =
      await tf.setdiff1dAsync(x, y);
    expect(out).to.lookLike(expectedOut);
    expect(indices).to.lookLike(expectedIndices);
  });
  it("  -- error if x or y are not 1d", async () => {
    const x = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const y = [1, 3, 5];
    // const expectedOut = [2, 4, 6];
    // const expectedIndices = [1, 3, 5];
    await expect(tf.setdiff1dAsync(x, y)).to.be.rejectedWith(
      `x should be 1D tensor, but got x (2,3).`
    );
  });
}
