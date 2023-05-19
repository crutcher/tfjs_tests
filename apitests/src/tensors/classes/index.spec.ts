import * as chai from "chai";
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import spies from "chai-spies";
chai.use(spies);
import * as Tensor from "./Tensor";

/* ---- Creating Tensors: tf classes related to tensors ---- */
describe("**** TENSORS: Classes ****", () => {
  /* ---- tf.Tensor class---- *
  A tf.Tensor object represents an immutable, multidimensional array of numbers that has a shape and a data type.
  */
  describe("tf.Tensor(start, stop, num):", Tensor.run.bind(this));
});
