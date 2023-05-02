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
  it("  -- default options", () => {
    const t: tfTypes.Scalar = tf.scalar(2);
    expect(t.dtype).to.equal("float32");
    expect(t).to.haveShape([]);
    expect(t.arraySync()).to.eql(2);
  });
  it("  -- shapes", () => {
    const t: tfTypes.Scalar = tf.scalar(true, "bool");
    expect(t).to.haveDtype("bool");
    expect(t.shape).to.eql([]);
    // Note: arraySync converts bools to numbers (0 or 1)
    expect(t.arraySync()).to.eql(1);
  });
}
