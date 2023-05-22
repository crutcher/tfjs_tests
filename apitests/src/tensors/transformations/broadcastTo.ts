import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/* CONSTANTS: */
const DEFAULT_ARR = [1, 2, 3];
const DEFAULT_BROADCAST_SHAPE = [2, 3];
const DEFAULT_EXPECTED_RESULT = [
  [1, 2, 3],
  [1, 2, 3],
];

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.broadcastTo (x, shape)-- */
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
  it("  -- array=>tensor : default example", () => {
    const t: tfTypes.Tensor = tf.broadcastTo(
      DEFAULT_ARR,
      DEFAULT_BROADCAST_SHAPE
    );
    expect(t).to.lookLike(DEFAULT_EXPECTED_RESULT);
  });
  it("  -- array=>tensor : incompatible shape - number of columns must match", () => {
    const arr = [1, 1, 1, 1]; // shape = [4]
    expect(() => tf.broadcastTo(arr, DEFAULT_BROADCAST_SHAPE)).to.throw(
      `broadcastTo(): [4] cannot be broadcast to [2,3].`
    );
  });
  it("  -- tensor=>tensor : default example", () => {
    const x: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor(DEFAULT_ARR);
    const t: tfTypes.Tensor = tf.broadcastTo(x, DEFAULT_BROADCAST_SHAPE);
    expect(t).to.lookLike(DEFAULT_EXPECTED_RESULT);
  });
  it("  -- 2d example : shape [3, 3] => [2, 3, 3] ", () => {
    const arr = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const broadcastShape = [2, 3, 3];
    const expectedResult = [
      [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],

      [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],
    ];
    const x: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor(arr);
    const t: tfTypes.Tensor = tf.broadcastTo(x, broadcastShape);
    expect(t).to.lookLike(expectedResult);
  });
  it("  -- if array has shape if smaller dimension, it will copy that value : shape [2, 1] => [2, 2] ", () => {
    const arr = [[5], [6]];
    const broadcastShape = [2, 2];
    const expectedResult = [
      [5, 5],
      [6, 6],
    ];
    const x: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor(arr);
    const t: tfTypes.Tensor = tf.broadcastTo(x, broadcastShape);
    expect(t).to.lookLike(expectedResult);
  });
}
