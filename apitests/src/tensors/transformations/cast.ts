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
  it("  -- float to int ", () => {
    const x: tfTypes.Tensor1D = tf.tensor1d([1.5, 2.5, 3]);
    const y: tfTypes.Tensor1D = tf.cast(x, "int32");
    const expected = [1, 2, 3];
    expect(y).to.haveDtype("int32");
    expect(y).to.lookLike(expected);
  });
  it("  -- float to bool ", () => {
    // 0 gets cast to false, everything else to true
    const x: tfTypes.Tensor2D = tf.tensor2d([1.5, 2.5, 0, 3], [2, 2]);
    const y: tfTypes.Tensor1D = tf.cast(x, "bool");
    const expected = [
      [1, 1],
      [0, 1],
    ];
    expect(y).to.haveDtype("bool");
    expect(y.arraySync()).to.eql(expected);
  });
}
