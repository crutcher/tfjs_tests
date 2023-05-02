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
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    expect(t.dtype).to.equal("float32");
  });
  it("  -- shapes", () => {
    const t: tfTypes.Tensor<tfTypes.Rank.R2> = tf.tensor([2, 3, 4, 5], [2, 2]);
    expect(t).to.haveDtype("float32");
    expect(t).to.haveShape([2, 2]);
    expect(t.arraySync()).to.eql([
      [2.0, 3.0],
      [4.0, 5.0],
    ]);
  });
  it("  -- dtypes", () => {
    const t: tfTypes.Tensor<tfTypes.Rank.R2> = tf.tensor(
      [
        [2, 3],
        [4, 5],
      ],
      undefined,
      "int32"
    );
    expect(t).to.haveDtype("int32");
    expect(t).to.haveShape([2, 2]);
    expect(t.arraySync()).to.eql([
      [2, 3],
      [4, 5],
    ]);
  });
}