import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.tile (x, reps) */
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
    const reps: number[] = [4];
    const a: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3]);
    const expectedTile = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
    const expectedShape = [12]; // [3 * 4]
    // assertions:
    const result: tfTypes.Tensor1D = a.tile(reps);
    expect(result).to.lookLike(expectedTile);
    expect(result).to.haveShape(expectedShape);
  });

  it("  -- !! bad 1d example: negative rep", async () => {
    const reps: number[] = [-1];
    const a: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3]);
    // assertions:
    expect(() => a.tile(reps)).to.throw(
      `Tensor must have a shape comprised of positive integers but got shape [-3].`
    );
  });

  it("  -- !! bad 1d example: non-integer rep", async () => {
    const reps: number[] = [3.5];
    const a: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3]);
    // assertions:
    expect(() => a.tile(reps)).to.throw(
      `ensor must have a shape comprised of positive integers but got shape [10.5].`
    );
  });

  it("  -- 3d example: basic", async () => {
    const reps: number[] = [1, 2, 3];
    const a: tfTypes.Tensor3D = tf.tensor3d([
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
    const expectedTile = [
      [
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
      ],

      [
        [5, 6, 5, 6, 5, 6],
        [7, 8, 7, 8, 7, 8],
        [5, 6, 5, 6, 5, 6],
        [7, 8, 7, 8, 7, 8],
      ],

      [
        [9, 10, 9, 10, 9, 10],
        [11, 12, 11, 12, 11, 12],
        [9, 10, 9, 10, 9, 10],
        [11, 12, 11, 12, 11, 12],
      ],
    ];
    const expectedShape = [3, 4, 6]; // [3 * 1, 2 * 2, 2 * 3]
    // assertions:
    const result: tfTypes.Tensor3D = a.tile(reps);
    expect(result).to.lookLike(expectedTile);
    expect(result).to.haveShape(expectedShape);
  });

  it("  -- 3d example: set any dimension to 0", async () => {
    // tensor will be empty but shape will be duplicated as expected
    // array representation will be 1d empty array
    // string representation will show empty dimensions
    const reps: number[] = [1, 0, 2];
    const a: tfTypes.Tensor3D = tf.tensor3d([
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
    const expectedTile: number[] = [];
    const expectedShape = [3, 0, 4];
    const expectedSize = 0;
    // assertions:
    const result = a.tile(reps);
    expect(result).to.lookLike(expectedTile);
    expect(result).to.haveShape(expectedShape);
    expect(result).to.haveSize(expectedSize);
  });
}
