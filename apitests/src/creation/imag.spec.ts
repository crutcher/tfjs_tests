import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import * as loader from "../load-tf";
import type tfTypes from "@tensorflow/tfjs-core";

/* ---- tf.diag ---- *
  Returns the imaginary part of a complex (or real) tensor.
*/
describe("tf.imag(complexTensor): ", () => {
  // CONSTANTS
  const EXPECTED = [4.75, 5.75];

  // TESTS
  it("  -- basic", async () => {
    const tf: loader.TFModule = await loader.load();

    const COMPLEX_TENSOR: tfTypes.Tensor = tf.complex(
      [-2.25, 3.25],
      [4.75, 5.75]
    );
    const x: tfTypes.Tensor = COMPLEX_TENSOR;
    const t: tfTypes.Tensor = tf.imag(x);
    expect(t).to.haveShape([2]);
    expect(t).to.lookLike(EXPECTED);
  });
});
