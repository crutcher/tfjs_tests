//Chai + chai plugins
import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
//tensorflow + tensorflow dynamic loader
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../load-tf";

let tf: loader.TFModule;

export function run() {
  // CONSTANTS
  const START = 0;
  const STOP = 9;
  const STEP = 2;
  const DEFAULT_RESULT = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const STEP_RESULT = [0, 2, 4, 6, 8];

  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  it("  -- default", () => {
    const t: tfTypes.Tensor = tf.range(START, STOP);
    expect(t).to.haveShape([DEFAULT_RESULT.length]);
    expect(t).to.haveSize(DEFAULT_RESULT.length);
    expect(t).to.lookLike(DEFAULT_RESULT);
  });
  it("  -- step", () => {
    const t: tfTypes.Tensor = tf.range(START, STOP, STEP);
    expect(t).to.haveShape([STEP_RESULT.length]);
    expect(t).to.haveSize(STEP_RESULT.length);
    expect(t).to.lookLike(STEP_RESULT);
  });
  it("  -- dtype", () => {
    const t: tfTypes.Tensor = tf.range(START, STOP, STEP, "int32");
    expect(t).to.haveDtype("int32");
  });
}
