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
    const buffer: tfTypes.TensorBuffer<tfTypes.Rank.R2> = tf.buffer([2, 2]);
    expect(buffer).to.haveShape([2, 2]);
    expect(buffer.size).to.eql(4);
    buffer.set(4, 0, 0);
    buffer.set(6, 0, 1);
    buffer.set(8, 1, 0);
    buffer.set(10, 1, 1);
    const expected = [
      [4, 6],
      [8, 10],
    ];
    expect(buffer.toTensor().arraySync()).to.eql(expected);
  });
  it("  -- dtype", () => {
    const buffer: tfTypes.TensorBuffer<tfTypes.Rank.R2, "int32"> = tf.buffer(
      [2, 2],
      "int32"
    );
    expect(buffer).to.haveDtype("int32");
  });
  it("  -- values", () => {
    const buffer: tfTypes.TensorBuffer<
      tfTypes.Rank.R2,
      keyof tfTypes.DataTypeMap
    > = tf.buffer([2, 2], undefined, Int32Array.from([1, 2, 3, 4]));
    const expected = [
      [1, 2],
      [3, 4],
    ];
    expect(buffer.toTensor().arraySync()).to.eql(expected);
  });
}
