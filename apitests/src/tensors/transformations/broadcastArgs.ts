import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/* CONSTANTS: */
const EXAMPLES_2D = [
  {
    s0: [2, 2],
    s1: [2, 1],
    expectedShape: [2, 2],
  },
  {
    s0: [1, 2],
    s1: [2, 2],
    expectedShape: [2, 2],
  },
  {
    s0: [1, 2],
    s1: [2, 1],
    expectedShape: [2, 2],
  },
];

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.broadcastArgs (s0, s1)--
  Return the shape of s0 op s1 with broadcast.
  s0 and s1 represent shapes
 */
export function run() {
  /* HOOKS: */
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });

  /* TESTS: */
  it("  -- 2d examples: [2, n], [n, 2] => [2, 2]; n∈{1, 2}", () => {
    EXAMPLES_2D.forEach(({ s0, s1, expectedShape }) => {
      const t: tfTypes.Tensor = tf.broadcastArgs(s0, s1);
      expect(t).to.lookLike(expectedShape);
    });
  });
  it("  -- bad 2d example: n >= 3", () => {
    const s0 = [1, 2];
    const s1 = [2, 3];
    expect(() => tf.broadcastArgs(s0, s1)).to.throw(
      `Operands could not be broadcast together with shapes 1,2 and 2,3.`
    );
  });
  it("  -- 2d example (passing but bad): if s0, s1 are floats, they are truncated to ints", () => {
    const s0 = [1.2, 2.5];
    const s1 = [2.3, 1];
    const expectedShape = [2, 2];
    const t: tfTypes.Tensor = tf.broadcastArgs(s0, s1);
    expect(t).to.lookLike(expectedShape);
  });
  it("  -- 3d example: [2, 3], [3, 2, 3] => [2, 2]", () => {
    const s0 = [2, 3];
    const s1 = [3, 2, 3];
    const expectedShape = [3, 2, 3];
    const t: tfTypes.Tensor = tf.broadcastArgs(s0, s1);
    expect(t).to.lookLike(expectedShape);
  });
}
