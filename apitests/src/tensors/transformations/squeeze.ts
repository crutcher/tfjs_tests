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

/* -- tf.squeeze (x, axis?) */
export function run() {
  /* CONSTANTS: */
  const INITIAL_TENSOR_VALS = [
    [[[[1], [2], [3], [4]]], [[[5], [6], [7], [8]]]],
  ];
  // to inialize before each test
  let x: tfTypes.Tensor5D;

  /* HOOKS: */
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  beforeEach(() => {
    x = tf.tensor5d(INITIAL_TENSOR_VALS);
  });

  /* TESTS: */

  it("  -- default", async () => {
    const expectedResult = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ];
    const expectedShape = [2, 4];
    // assertions:
    const y = x.squeeze();
    expect(y).to.haveShape(expectedShape);
    expect(y).to.lookLike(expectedResult);
  });

  it("  -- specify dimensions", async () => {
    const axis = [0, 4];
    const expectedResult = [[[1, 2, 3, 4]], [[5, 6, 7, 8]]];
    const expectedShape = [2, 1, 4];
    // assertions:
    const y = x.squeeze(axis);
    expect(y).to.haveShape(expectedShape);
    expect(y).to.lookLike(expectedResult);
  });
  it("  -- error : dimensions aren't of size 1", async () => {
    const axis = [1];
    // assertions:
    expect(() => x.squeeze(axis)).to.throw(
      `Can't squeeze axis 1 since its dim '2' is not 1`
    );
  });
  it("  -- error : dimension out of range", async () => {
    const axis = [5];
    // assertions:
    expect(() => x.squeeze(axis)).to.throw(
      `All values in axis param must be in range [-5, 5) but got axis 5`
    );
  });
}
