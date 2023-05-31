import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.gather (x, indices, axis?, batchDims?) */
export function run() {
  /* CONSTANTS: */
  const DEFAULT_INDICES = [1, 3, 3];
  const DEFAULT_INDICES_FOR_AXIS_EXAMPLES = [0, 1, 1];

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
    const indicesAsArr = DEFAULT_INDICES;
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    const indices: tfTypes.Tensor1D = tf.tensor1d(indicesAsArr, "int32");
    const expectedResult = [2, 4, 4];
    // assertions:
    const result = x.gather(indices);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- error: indices not of type integer", async () => {
    const indicesAsArr = DEFAULT_INDICES;
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    const indices: tfTypes.Tensor1D = tf.tensor1d(indicesAsArr);
    // assertions:
    expect(() => x.gather(indices)).to.throw(
      `Argument 'indices' passed to 'gather' must be int32 tensor, but got float32 tensor`
    );
  });

  it("  -- error: indices out of range", async () => {
    const indicesAsArr = [20, 1];
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    const indices: tfTypes.Tensor1D = tf.tensor1d(indicesAsArr, "int32");
    // assertions:
    expect(() => x.gather(indices)).to.throw(
      `GatherV2: the index value 20 is not in [0, 3]`
    );
  });

  it("  -- 2d example: basic", async () => {
    const indicesAsArr = DEFAULT_INDICES;
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
    ]);
    const indices: tfTypes.Tensor1D = tf.tensor1d(indicesAsArr, "int32");
    const expectedResult = [
      [3, 4],
      [7, 8],
      [7, 8],
    ];
    // assertions:
    const result = x.gather(indices);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- 2d example: basic", async () => {
    const indicesAsArr = DEFAULT_INDICES;
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
    ]);
    const indices: tfTypes.Tensor1D = tf.tensor1d(indicesAsArr, "int32");
    const expectedResult = [
      [3, 4],
      [7, 8],
      [7, 8],
    ];
    // assertions:
    const result = x.gather(indices);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- 2d example: 2d indices tensor", async () => {
    const indicesAsArr = [
      [0, 1],
      [3, 3],
      [4, 2],
    ];
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
    ]);
    const indices: tfTypes.Tensor2D = tf.tensor2d(
      indicesAsArr,
      undefined,
      "int32"
    );
    const expectedResult = [
      [
        [1, 2],
        [3, 4],
      ],

      [
        [7, 8],
        [7, 8],
      ],

      [
        [9, 10],
        [5, 6],
      ],
    ];
    // assertions:
    const result = x.gather(indices);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- axis example (2d): basic", async () => {
    const axis = 1;
    const indicesAsArr = DEFAULT_INDICES_FOR_AXIS_EXAMPLES;
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const indices: tfTypes.Tensor1D = tf.tensor1d(indicesAsArr, "int32");
    const expectedResult = [
      [1, 2, 2],
      [3, 4, 4],
      [5, 6, 6],
    ];
    // assertions:
    const result = x.gather(indices, axis);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- 2d example: with batch dims", async () => {
    const batchDims = 1;
    const axis = 1;
    const indicesAsArr = DEFAULT_INDICES_FOR_AXIS_EXAMPLES;
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const indices: tfTypes.Tensor1D = tf.tensor1d(indicesAsArr, "int32");
    const expectedResult = [1, 4, 6];
    // assertions:
    const result = x.gather(indices, axis, 1);
    expect(result).to.lookLike(expectedResult);
  });

  it("  -- error: 2d example: with bad batch dims", async () => {
    const batchDims = 1;
    const indicesAsArr = DEFAULT_INDICES_FOR_AXIS_EXAMPLES;
    const axis = 1;
    const x: tfTypes.Tensor2D = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
      // ! adding these extra rows will cause an error !
      [7, 8],
      [9, 10],
    ]);
    const indices: tfTypes.Tensor1D = tf.tensor1d(indicesAsArr, "int32");
    // assertions:
    expect(() => x.gather(indices, axis, batchDims)).to.throw(
      `x.shape[0]: 5 should be equal to indices.shape[0]: 3.`
    );
  });
}
