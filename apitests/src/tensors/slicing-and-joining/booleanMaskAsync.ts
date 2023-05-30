import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import { default as chaiAsPromised } from "chai-as-promised";
chai.use(chaiAsPromised);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.booleanMaskAsync (tensor, mask, axis?) */
export function run() {
  /* CONSTANTS: */
  const INITIAL_TENSOR_VALS = [
    [1, 2],
    [3, 4],
    [5, 6],
  ];
  // to inialize before each test
  let x: tfTypes.Tensor2D;
  /* HOOKS: */
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  beforeEach(() => {
    x = tf.tensor2d(INITIAL_TENSOR_VALS);
  });
  /* TESTS: */

  it("  -- basic example: 2d tensor with 1d mask", async () => {
    const mask = tf.tensor1d([1, 0, 1], "bool");
    const expectedValues = [
      [1, 2],
      [5, 6],
    ];
    const result = await tf.booleanMaskAsync(x, mask);
    // assertions:
    expect(result).to.lookLike(expectedValues);
  });
  it("  -- example with same result: mask contains non-binary values", async () => {
    const mask = tf.tensor1d([5, 0, 8], "bool");
    const expectedValues = [
      [1, 2],
      [5, 6],
    ];
    const result = await tf.booleanMaskAsync(x, mask);
    // assertions:
    expect(result).to.lookLike(expectedValues);
  });
  it("  -- error: mask is of type float (but contains only 1 and 0)", async () => {
    const mask = tf.tensor1d([1, 0, 1], "float32");
    // assertions:
    await expect(tf.booleanMaskAsync(x, mask)).to.be.rejectedWith(
      `Argument 'mask' passed to 'boolMask' must be bool tensor, but got float32 tensor`
    );
  });
  it("  -- error: mask dimensions don't fit tensor", async () => {
    const mask = tf.tensor1d([1, 0, 1, 1], "bool");
    // assertions:
    await expect(tf.booleanMaskAsync(x, mask)).to.be.rejectedWith(
      `mask's shape must match the first K dimensions of tensor's shape, Shapes 3 and 4 must match`
    );
  });
  it("  -- 3d tensor with 2d mask: results in 2d tensor", async () => {
    const y: tfTypes.Tensor3D = tf.tensor([
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
    ]);
    const mask = tf.tensor(
      [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      undefined,
      "bool"
    );
    const expectedValues = [
      [1, 2],
      [7, 8],
      [9, 10],
      [11, 12],
    ];
    const expectedShape = [4, 2];
    // there are 4 "true"s in the mask,
    // so the first dimension of the result is 4
    const result = await tf.booleanMaskAsync(y, mask);
    // assertions:
    expect(result).to.lookLike(expectedValues);
    expect(result).to.haveShape(expectedShape);
  });
}
