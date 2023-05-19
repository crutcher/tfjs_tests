//Chai + chai plugins
import * as chai from "chai";
const expect = chai.expect;
import spies from "chai-spies";
chai.use(spies);
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import * as sinon from "sinon";
//tensorflow + tensorflow dynamic loader
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

let tf: loader.TFModule;

export function run() {
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  it("  -- default", () => {
    const t: tfTypes.Tensor = tf.tensor([2, 3]);
    const stub = sinon.stub(t, "print").callsFake((verbose = false): string => {
      return t.toString(verbose);
    });
    const output = t.print();
    stub.restore();
    expect(output).to.eql("Tensor\n" + "    [2, 3]");
  });
  it("  -- verbose", () => {
    const t: tfTypes.Tensor = tf.tensor([2, 3]);
    const stub = sinon.stub(t, "print").callsFake((verbose = false): string => {
      return t.toString(verbose);
    });
    const output = t.print(true);
    stub.restore();
    expect(output).to.eql(
      "Tensor\n" +
        "  dtype: float32\n" +
        "  rank: 1\n" +
        "  shape: [2]\n" +
        "  values:\n" +
        "    [2, 3]"
    );
  });
}
