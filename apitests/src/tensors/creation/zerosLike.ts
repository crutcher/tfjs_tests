import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

let tf: loader.TFModule;

/* ---- zerosLike ---- *
  Creates a tf.Tensor with all elements set to 0 with the same shape as the given tensor.
*/
export function run() {
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });

  // TESTS
  it("  -- default", async () => {
    const x: tfTypes.Tensor = tf.tensor([2, 2, 1]);
    const t: tfTypes.Tensor = tf.zerosLike(x);
    expect(t).to.haveShape(x.shape);
    expect(t).to.haveSize(x.size);
    expect(t).to.be.allZeros;
  });
}
