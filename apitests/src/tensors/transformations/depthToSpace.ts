import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.depthToSpace (x, blockSize, dataFormat?)-- */
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

  it("  -- shape [1, 1, 1, 4], blockSize 2 => [1, 2, 2, 1]", () => {
    const x: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
    const blockSize = 2;
    const expected = [
      [
        [[1], [2]],

        [[3], [4]],
      ],
    ];
    const y: tfTypes.Tensor4D = tf.depthToSpace(x, blockSize);
    expect(y).to.haveShape([1, 2, 2, 1]);
    expect(y).to.lookLike(expected);
  });
  it("  -- shape [1, 1, 1, 9] blockSize 3 => [1, 2, 2, 1]", () => {
    const x: tfTypes.Tensor4D = tf.tensor4d(
      [1, 2, 3, 4, 5, 6, 7, 8, 9],
      [1, 1, 1, 9]
    );
    const blockSize = 3;
    const expected = [
      [
        [[1], [2], [3]],

        [[4], [5], [6]],

        [[7], [8], [9]],
      ],
    ];
    const y: tfTypes.Tensor4D = tf.depthToSpace(x, blockSize);
    expect(y).to.haveShape([1, 3, 3, 1]);
    expect(y).to.lookLike(expected);
  });
  it("  -- error : dimension size not divisible by block size", () => {
    const x: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
    const blockSize = 3;
    expect(() => tf.depthToSpace(x, blockSize)).to.throw(
      `Dimension size must be evenly divisible by 9 but is 4 for depthToSpace with input shape 1,1,1,4`
    );
  });
}
