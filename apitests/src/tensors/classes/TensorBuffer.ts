import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/* CONSTANTS: */
const SHAPE: [number, number] = [2, 2];

/**** ---- MOCHA TEST FUNCTION: ---- *****/

/* -- tf.TensorBuffer class methods-- */
export function run() {
  /* HOOKS: */
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  // - to initialize before each test:
  let buffer: tfTypes.TensorBuffer<tfTypes.Rank.R2>;
  beforeEach(() => {
    buffer = tf.buffer(SHAPE);
  });

  /* TESTS: */
  it("  -- TensorBuffer.set()", () => {
    buffer.set(4, 0, 1);
    const t: tfTypes.Tensor<tfTypes.Rank.R2> = buffer.toTensor();
    const expected = [
      [0, 4],
      [0, 0],
    ];
    expect(t.arraySync()).to.eql(expected);
  });
  it("  -- TensorBuffer.set() : indices out of range", () => {
    // does not throw an error, but does not set the value either
    buffer.set(4, 0, 6);
    const t: tfTypes.Tensor<tfTypes.Rank.R2> = buffer.toTensor();
    const expected = [
      [0, 0],
      [0, 0],
    ];
    expect(t.arraySync()).to.eql(expected);
  });
  it("  -- TensorBuffer.set() : location dimension not represented by shape of buffer", () => {
    const NUM_COORDS = 3;
    const RANK = buffer.rank;
    expect(() => buffer.set(4, 0, 0, 1)).to.throw(
      `The number of provided coordinates (${NUM_COORDS}) must match the rank (${RANK})`
    );
  });
}
