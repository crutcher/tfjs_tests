//Chai + chai plugins
import * as chai from "chai";
const expect = chai.expect;
import spies from "chai-spies";
chai.use(spies);
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
//tensorflow + tensorflow dynamic loader
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
  it("  -- default", () => {
    const t: tfTypes.Tensor = tf.tensor(["ts.print()", "ts.print()"]);
    const spy = chai.spy.on(t, "print");
    t.print();
    expect(spy).to.have.been.called();
  });
  it("  -- verbose", () => {
    const verbose = true;
    const t: tfTypes.Tensor = tf.tensor(["ts.print()"]);
    const spy = chai.spy.on(t, "print");
    t.print(verbose);
    expect(spy).to.have.been.called();
  });
}
