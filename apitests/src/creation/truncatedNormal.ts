import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../load-tf";

let tf: loader.TFModule;

export function run() {
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  // CONSTANTS
  const SHAPE = [2, 2];
  const RANGE_START = -2;
  const RANGE_END = 2;
  const MEAN = 4;
  const STDDEV = 3;

  // TESTS
  it("  -- default", async () => {
    const t: tfTypes.Tensor = tf.truncatedNormal(SHAPE);
    expect(t).to.haveShape(SHAPE);
    expect(t).to.have.allValuesInRange(RANGE_START, RANGE_END);
  });
  it("  -- mean", async () => {
    const t: tfTypes.Tensor = tf.truncatedNormal(SHAPE, MEAN);
    const START = MEAN - 2;
    const END = MEAN + 2;
    expect(t).to.have.allValuesInRange(START, END);
  });
  it("  -- stdDev", async () => {
    const t: tfTypes.Tensor = tf.truncatedNormal(SHAPE, undefined, STDDEV);
    const START = RANGE_START * STDDEV;
    const END = RANGE_END * STDDEV;
    expect(t).to.have.allValuesInRange(START, END);
  });
}
