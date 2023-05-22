import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";
// utils
import { areEqual } from "../../utils/tensor-utils";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/* CONSTANTS: */
const BLOCK_SHAPE = [2, 2];
const CROPS = [
  [0, 0],
  [0, 0],
];

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.batchToSpaceND(x, blockShape, crops)-- */
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

  // the innermost dimension is unchanged
  // the middle dimensions get modified by the blockshape
  // the rannk of the tensor remains unchanged
  // the size of the tensor remains unchanged

  it("  -- default example: shape [4, 1, 1, 1] --> [1, 2, 2, 1]", () => {
    const t: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
    const expectedShape = [1, 2, 2, 1];
    const expectedResult = [
      [
        [[1], [2]],
        [[3], [4]],
      ],
    ];
    const x: tfTypes.Tensor = t.batchToSpaceND(BLOCK_SHAPE, CROPS);
    expect(x).to.haveShape(expectedShape);
    expect(x).to.lookLike(expectedResult);
  });
  it("  -- shape [4, 1, 1, 3] --> [1, 2, 2, 3]", () => {
    const t: tfTypes.Tensor4D = tf.tensor4d(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      [4, 1, 1, 3]
    );
    const expectedShape = [1, 2, 2, 3];
    const expectedResult = [
      [
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [
          [7, 8, 9],
          [10, 11, 12],
        ],
      ],
    ];
    const x: tfTypes.Tensor = t.batchToSpaceND(BLOCK_SHAPE, CROPS);
    expect(x).to.haveShape(expectedShape);
    expect(x).to.lookLike(expectedResult);
  });
  it("  -- default : bad blockShape - not evenly divisible", () => {
    const t: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
    const badBlockShape = [2, 3];
    expect(() => t.batchToSpaceND(badBlockShape, CROPS)).to.throw(
      `input tensor batch is 4 but is not divisible by the product of the elements of blockShape 2 * 3 === 6`
    );
  });
  it("  -- default: blockShape that returns original tensor", () => {
    const t: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
    const x: tfTypes.Tensor = t.batchToSpaceND(BLOCK_SHAPE, CROPS);
    expect(areEqual(x, t)).to.be.true;
  });
  it("  -- shape [8, 1, 3, 1] --> [2, 2, 6, 1]", () => {
    const t: tfTypes.Tensor4D = tf.tensor4d(
      [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24,
      ],
      [8, 1, 3, 1]
    );
    const expectedShape = [2, 2, 6, 1];
    const expectedResult = [
      [
        [[1], [7], [2], [8], [3], [9]],

        [[13], [19], [14], [20], [15], [21]],
      ],

      [
        [[4], [10], [5], [11], [6], [12]],

        [[16], [22], [17], [23], [18], [24]],
      ],
    ];
    const x: tfTypes.Tensor = t.batchToSpaceND(BLOCK_SHAPE, CROPS);
    expect(x).to.haveShape(expectedShape);
    expect(x).to.lookLike(expectedResult);
  });
  it("  -- bad crops : offest results in null tensor", () => {
    const t: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
    const badCrops = [
      [1, 1],
      [1, 1],
    ];
    // shape has a 0 in 1st dimension, so first dimension will be null
    // and will have no further dimensions or values
    // ie. will result in empy array []
    const expectedShape = [1, 0, 0, 1];
    const expectedResult: number[] = [];
    const x: tfTypes.Tensor = t.batchToSpaceND(BLOCK_SHAPE, badCrops);
    expect(x).to.haveShape(expectedShape);
    expect(x).to.lookLike(expectedResult);
  });
  it("  -- bad crops : crops of wrong shape", () => {
    const t: tfTypes.Tensor4D = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
    // inner dimension should have length of 2 to reflect shape of tensor
    const badCrops = [[0], [0]];
    expect(() => t.batchToSpaceND(BLOCK_SHAPE, badCrops)).to.throw(
      `Negative size values should be exactly -1 but got NaN for the slice() size at index 1.`
    );
  });
  it("  -- with 2d tensor: blockshape = [2]; shape [2, 2] => [1, 4]", () => {
    const t: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const newBlockShape = [2];
    const newCrops = [[0, 0]];
    const expectedShape = [1, 4];
    const expectedResult = [[1, 3, 2, 4]]; //NB: this is different
    const x: tfTypes.Tensor = t.batchToSpaceND(newBlockShape, newCrops);
    expect(x).to.haveShape(expectedShape);
    expect(x).to.lookLike(expectedResult);
  });
}
