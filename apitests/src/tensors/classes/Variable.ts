import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";
// Types
import { TFArray } from "../../utils/tensor-utils";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.Variable class methods-- */
export function run() {
  /* CONSTANTS: */
  const INITIAL = [1, 2, 3];
  const NEW_VALUES = [4, 5, 6];
  const BAD_NEW_VALUES = [
    [7, 8],
    [9, 10],
  ];

  /* HOOKS: */
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  // - to initialize before each test:
  let t: tfTypes.Tensor;
  let tShape: TFArray;
  let x: tfTypes.Variable;
  beforeEach(() => {
    t = tf.tensor(INITIAL);
    tShape = t.shape;
    x = tf.variable(t);
  });

  /* TESTS: */
  it("  -- Variable.assign()", () => {
    /*
    Assign a new tf.Tensor to this variable.
    The new tf.Tensor must have the same shape and dtype as the old tf.Tensor.
    */
    const t2: tfTypes.Tensor = tf.tensor(NEW_VALUES);
    x.assign(t2);
    expect(x).to.haveShape(t2.shape);
    expect(x).to.lookLike(NEW_VALUES);
  });
  it("  -- Variable.assign() : bad shape", () => {
    /*
    The new tensor does not have the same shape and dtype as the old tf.Tensor.
    */
    const t2: tfTypes.Tensor = tf.tensor(BAD_NEW_VALUES);
    expect(() => x.assign(t2)).to.throw(
      `shape of the new value (${t2.shape}) and previous value (${tShape}) must match`
    );
  });
}
