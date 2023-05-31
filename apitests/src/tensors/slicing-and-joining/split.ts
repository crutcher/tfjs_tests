import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.split (x, numOrSizeSplits, axis?) */
export function run() {
  /* CONSTANTS: */
  const DEFAULT_INITIAL_VALUES = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
  ];

  /* HOOKS: */
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });

  /* TESTS: */

  it("  -- specifying the number of segments: passing", async () => {
    const numSegments = 4;
    const x: tfTypes.Tensor2D = tf.tensor2d(DEFAULT_INITIAL_VALUES);
    const expectedShape = [1, 4];
    const aExpectedValues = [[1, 2, 3, 4]];
    const bExpectedValues = [[5, 6, 7, 8]];
    const cExpectedValues = [[9, 10, 11, 12]];
    const dExpectedValues = [[13, 14, 15, 16]];
    // assertions:
    const [a, b, c, d]: tfTypes.Tensor2D[] = tf.split(x, numSegments);
    expect(a).to.haveShape(expectedShape);
    expect(a).to.lookLike(aExpectedValues);
    expect(b).to.lookLike(bExpectedValues);
    expect(c).to.lookLike(cExpectedValues);
    expect(d).to.lookLike(dExpectedValues);
  });

  it("  -- specifying the number of segments: with axis", async () => {
    const axis = 1;
    const numSegments = 4;
    const x: tfTypes.Tensor2D = tf.tensor2d(DEFAULT_INITIAL_VALUES);
    const expectedShape = [4, 1];
    const aExpectedValues = [[1], [5], [9], [13]];
    const bExpectedValues = [[2], [6], [10], [14]];
    const cExpectedValues = [[3], [7], [11], [15]];
    const dExpectedValues = [[4], [8], [12], [16]];
    // assertions:
    const [a, b, c, d]: tfTypes.Tensor2D[] = tf.split(x, numSegments, axis);
    expect(a).to.haveShape(expectedShape);
    expect(a).to.lookLike(aExpectedValues);
    expect(b).to.lookLike(bExpectedValues);
    expect(c).to.lookLike(cExpectedValues);
    expect(d).to.lookLike(dExpectedValues);
  });

  it("  -- !! specifying the number of segments: bad values", async () => {
    const numSegments = 3;
    const x: tfTypes.Tensor2D = tf.tensor2d(DEFAULT_INITIAL_VALUES);
    // assertions:
    expect(() => tf.split(x, numSegments)).to.throw(
      `Number of splits must evenly divide the axis.`
    );
  });

  it("  -- specifying the size of each segment: passing", async () => {
    const segmentSize = [1, 2, 1];
    const x: tfTypes.Tensor2D = tf.tensor2d(DEFAULT_INITIAL_VALUES);
    const expectedResults = [
      [[1, 2, 3, 4]],
      [
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ],
      [[13, 14, 15, 16]],
    ];
    // assertions:
    const tensors: tfTypes.Tensor2D[] = tf.split(x, segmentSize);
    expect(tensors).to.have.lengthOf(3);
    tensors.forEach((tensor, i) => {
      expect(tensor).to.lookLike(expectedResults[i]);
    });
  });
  it("  -- !! specifying the size of each segment: bad values", async () => {
    const segmentSize = [1, 2];
    const x: tfTypes.Tensor2D = tf.tensor2d(DEFAULT_INITIAL_VALUES);
    // assertions:
    expect(() => tf.split(x, segmentSize)).to.throw(
      `The sum of sizes must match the size of the axis dimension.`
    );
  });
}
