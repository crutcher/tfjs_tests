import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.rand (shape, randFunction, dtype?) */
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

  it("  -- basic example", () => {
    const shape = [2, 3];
    const randFunction = () => Math.random();
    // assertions:
    const result: tfTypes.Tensor = tf.rand(shape, randFunction);
    expect(result).to.haveShape(shape);
    expect(result).to.have.allValuesInRange(0, 1);
  });
  it("  -- basic example: with dtype", () => {
    const shape = [2, 3];
    const randFunction = () => Math.random();
    const dtype = "int32";
    // assertions:
    const result: tfTypes.Tensor = tf.rand(shape, randFunction, dtype);
    expect(result).to.haveShape(shape);
    expect(result).to.have.allValuesInRange(0, 1);
    expect(result.dtype).to.equal(dtype);
  });
}
