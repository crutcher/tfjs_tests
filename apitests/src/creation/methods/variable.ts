import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

let tf: loader.TFModule;

/* ---- tf.variable(tensor) returns a Variable which is a mutable tensor ---- */
export function run() {
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  // CONSTANTS
  const INITIAL = [1, 2, 3];
  const NEW_VALUES = [4, 5, 6];

  const INVALID_SHAPE_VALUES = [4, 5, 6, 7, 8, 9];
  const INVALID_SHAPE_ERROR =
    "shape of the new value (6) and previous value (3) must match";
  const CUSTOM_NAME = "myName";

  const INVALID_DTYPE_VALUES = [true, true, false];
  const INVALID_DTYPE_ERROR =
    "dtype of the new value (bool) and previous value (float32) must match";

  // TESTS
  it("  -- default", async () => {
    const x: tfTypes.Tensor = tf.tensor(INITIAL);
    const t: tfTypes.Variable = tf.variable(x);
    const y: tfTypes.Tensor = tf.tensor(NEW_VALUES);
    t.assign(y);
    expect(t).to.haveShape(y.shape);
    expect(t).to.lookLike(NEW_VALUES);
  });
  it("  -- invalid shape", async () => {
    const x: tfTypes.Tensor = tf.tensor(INITIAL);
    const t: tfTypes.Variable = tf.variable(x);
    const y: tfTypes.Tensor = tf.tensor(INVALID_SHAPE_VALUES);
    expect(() => t.assign(y)).to.throw(INVALID_SHAPE_ERROR);
  });
  it("  -- invalid dtypes", async () => {
    const x: tfTypes.Tensor = tf.tensor(INITIAL);
    const t: tfTypes.Variable = tf.variable(x);
    const y: tfTypes.Tensor = tf.tensor(INVALID_DTYPE_VALUES);
    expect(() => t.assign(y)).to.throw(INVALID_DTYPE_ERROR);
  });
  it("  -- name", async () => {
    const x: tfTypes.Tensor = tf.tensor(INITIAL);
    const t: tfTypes.Variable = tf.variable(x, undefined, CUSTOM_NAME);
    const y: tfTypes.Tensor = tf.tensor(NEW_VALUES);
    t.assign(y);
    expect(t.name).to.equal(CUSTOM_NAME);
  });
}
