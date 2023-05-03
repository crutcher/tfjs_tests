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
  it("  -- conversion from floats", () => {
    const real: tfTypes.Tensor2D = tf.tensor2d(
      [
        [0, 1],
        [2, 3],
      ],
      undefined,
      "float32"
    );
    const imag: tfTypes.Tensor2D = tf.tensor2d(
      [
        [0, 10],
        [20, 30],
      ],
      undefined,
      "float32"
    );
    const complex: tfTypes.Tensor2D = tf.complex(real, imag);
    expect(complex).to.haveDtype("complex64");
    expect(complex.size).to.equal(4);

    expect(tf.real(complex).arraySync()).to.eql(real.arraySync());
    expect(tf.imag(complex).arraySync()).to.eql(imag.arraySync());
  });
  it("  -- conversion from ints", () => {
    const real: tfTypes.Tensor2D = tf.tensor2d(
      [
        [0, 1],
        [2, 3],
      ],
      undefined,
      "int32"
    );
    const imag: tfTypes.Tensor2D = tf.tensor2d(
      [
        [0, 10],
        [20, 30],
      ],
      undefined,
      "int32"
    );
    const complex: tfTypes.Tensor2D = tf.complex(real, imag);
    expect(complex).to.haveDtype("complex64");
    expect(complex.size).to.equal(4);

    expect(tf.real(complex).arraySync()).to.eql(real.arraySync());
    expect(tf.imag(complex).arraySync()).to.eql(imag.arraySync());
  });
  it("  -- bad input shapes", () => {
    const real: tfTypes.Tensor2D = tf.tensor2d([
      [0, 1],
      [2, 3],
    ]);
    const imag: tfTypes.Tensor2D = tf.tensor2d([[0, 10]]);
    expect(() => tf.complex(real, imag)).to.throw("must match");
  });
}
