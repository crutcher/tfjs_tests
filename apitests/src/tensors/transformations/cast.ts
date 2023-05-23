import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.cast (x, dtype)-- */
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
  it("  -- float to int", () => {
    const x: tfTypes.Tensor1D = tf.tensor1d([1.5, 2.5, 3]);
    const y: tfTypes.Tensor1D = tf.cast(x, "int32");
    const expected = [1, 2, 3];
    expect(y).to.haveDtype("int32");
    expect(y).to.lookLike(expected);
  });
  it("  -- float to bool", () => {
    // 0 gets cast to false, everything else to true
    const x: tfTypes.Tensor2D = tf.tensor2d([1.5, 2.5, 0, 3], [2, 2]);
    const y: tfTypes.Tensor2D = tf.cast(x, "bool");
    const expected = [
      [true, true],
      [false, true],
    ];
    expect(y).to.haveDtype("bool");
    expect(y).to.lookLike(expected);
  });
  it("  -- float to bool", () => {
    // 0 gets cast to false, everything else to true
    const x: tfTypes.Tensor3D = tf.tensor3d(
      [1.5, 2.5, 0, 3, 0.2, 0, 20, 0],
      [2, 2, 2]
    );
    const y: tfTypes.Tensor3D = tf.cast(x, "bool");
    const expected = [
      [
        [true, true],
        [false, true],
      ],

      [
        [true, false],
        [true, false],
      ],
    ];
    expect(y).to.haveDtype("bool");
    expect(y).to.lookLike(expected);
  });
  it("  -- float to string : throws error ", () => {
    const x: tfTypes.Tensor1D = tf.tensor1d([1.5, 2.5, 3]);
    expect(() => tf.cast(x, "string")).to.throw(
      `Only strings can be casted to strings`
    );
  });
  it("  -- int to float", () => {
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3], "int32");
    const y: tfTypes.Tensor1D = tf.cast(x, "float32");
    const expected = [1, 2, 3];
    expect(y).to.haveDtype("float32");
    expect(y).to.lookLike(expected);
  });
  it("  -- int to bool", () => {
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 0, 3], "int32");
    const y: tfTypes.Tensor1D = tf.cast(x, "bool");
    const expected = [true, false, true];
    expect(y).to.haveDtype("bool");
    expect(y).to.lookLike(expected);
  });
}
