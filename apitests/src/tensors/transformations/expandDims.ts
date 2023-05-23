import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.expandDims (x, axis?)-- */
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

  it("  -- 1d example", () => {
    const x: tfTypes.Tensor1D = tf.tensor1d([1, 2, 3, 4]);
    const expectedShape = [1, 4];
    const y: tfTypes.Tensor = x.expandDims();
    expect(y).to.haveShape(expectedShape);
  });
  it("  -- 2d example", () => {
    const initialShape: [number, number] = [2, 2];
    const expectedShape = [1, 2, 2];
    const x: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4], initialShape);
    const y: tfTypes.Tensor = x.expandDims();
    expect(y).to.haveShape(expectedShape);
  });
  it("  -- 2d example : with axis", () => {
    const axis = 1;
    const initialShape: [number, number] = [2, 2];
    const expectedShape = [2, 1, 2];
    const expectedValue = [[[1, 2]], [[3, 4]]];
    const x: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4], initialShape);
    const y: tfTypes.Tensor = x.expandDims(axis);
    expect(y).to.haveShape(expectedShape);
    expect(y).to.lookLike(expectedValue);
  });
  it("  -- bad 2d example : with axis out of range", () => {
    const axis = 3;
    const initialShape: [number, number] = [2, 2];
    const x: tfTypes.Tensor2D = tf.tensor2d([1, 2, 3, 4], initialShape);
    expect(() => x.expandDims(axis)).to.throw(
      `Axis must be <= rank of the tensor`
    );
  });
}
