import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

let tf: loader.TFModule;

export function run() {
  // CONSTANTS
  const TEST_ARRAY = [3, 2];

  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  it("  -- from tensor", () => {
    const x: tfTypes.Tensor = tf.tensor(TEST_ARRAY);
    const xShape = x.shape;
    const xSize = x.size;
    const t: tfTypes.Tensor = tf.onesLike(x);
    expect(t).to.haveShape(xShape);
    expect(t).to.haveSize(xSize);
    expect(t).to.be.allOnes;
  });
  it("  -- from typed array", () => {
    const typedArray = new Int32Array(TEST_ARRAY);
    const t: tfTypes.Tensor = tf.onesLike(typedArray);
    const shape = [TEST_ARRAY.length];
    expect(t).to.haveShape(shape);
    expect(t).to.haveSize(TEST_ARRAY.length);
    expect(t).to.be.allOnes;
  });
  it("  -- from empty array", () => {
    const typedArray = new Int32Array([]);
    const t: tfTypes.Tensor = tf.onesLike(typedArray);
    expect(t).to.haveShape([0]);
    expect(t).to.haveSize(0);
    expect(t).to.be.allOnes;
  });
}
