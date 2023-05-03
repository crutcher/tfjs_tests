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
  const IMAG = [-2.25, 3.25];
  const REAL = [4.75, 5.75];
  const EXPECTED = [4.75, 5.75];

  // TESTS
  it("  -- default", async () => {
    const x: tfTypes.Tensor = tf.complex(REAL, IMAG);
    const t: tfTypes.Tensor = tf.real(x);
    expect(t).to.haveShape([2]);
    expect(t).to.lookLike(EXPECTED);
  });
}
