import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";

let tf: loader.TFModule;

// Types
type TypedArray = Float32Array | Int32Array | Uint8Array | Uint16Array;

/* -- tf.Tensor class methods-- */
export function run() {
  before((done) => {
    loader.load().then((result: loader.TFModule) => {
      tf = result;
      // Complete the async stuff
      done();
    });
  });
  it("  -- Tensor.buffer()", async () => {
    // Returns a promise of tf.TensorBuffer that holds the underlying data.
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    try {
      const buffer: tfTypes.TensorBuffer<tfTypes.Rank.R1> = await t.buffer();
      expect(buffer).to.haveShape([2]);
      expect(buffer.size).to.eql(2);
      buffer.set(4, 0);
      expect(buffer.get(0)).to.eql(4);
      expect(buffer.toTensor()).to.lookLike([4, 3]);
    } catch (error) {
      let message = "";
      if (error instanceof Error) {
        message = error.message;
      }
      throw new Error(`Tensor.buffer(): could not create buffer: ${message}`);
    }
  });
  it("  -- Tensor.bufferSync()", () => {
    // Returns a tf.TensorBuffer that holds the underlying data.
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    const buffer: tfTypes.TensorBuffer<tfTypes.Rank.R1> = t.bufferSync();
    expect(buffer).to.haveShape([2]);
    expect(buffer.size).to.eql(2);
    buffer.set(4, 0);
    expect(buffer.get(0)).to.eql(4);
    expect(buffer.toTensor()).to.lookLike([4, 3]);
  });
  it("  -- Tensor.array()", async () => {
    // Returns the tensor data as a nested array. The transfer of data is done asynchronously.
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    try {
      const arr: number[] = await t.array();
      expect(arr).to.eql([2, 3]);
    } catch (error) {
      let message = "";
      if (error instanceof Error) {
        message = error.message;
      }
      throw new Error(
        `Tensor.array(): could not create array from Tensor object: ${message}`
      );
    }
  });
  it("  -- Tensor.arraySync()", () => {
    // Returns the tensor data as a nested array. The transfer of data is done synchronously.
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    const arr: number[] = t.arraySync();
    expect(arr).to.eql([2, 3]);
  });
  it("  -- Tensor.data()", async () => {
    /*
    Asynchronously downloads the values from the tf.Tensor.
    Returns a promise of TypedArray that resolves when the computation has finished.
    */
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    try {
      const data: TypedArray = await t.data();
      expect(data).to.eql(new Float32Array([2, 3]));
    } catch (error) {
      let message = "";
      if (error instanceof Error) {
        message = error.message;
      }
      throw new Error(`Tensor.buffer(): could not create buffer: ${message}`);
    }
  });
  it("  -- Tensor.dataToGPU()", () => {
    /*
    Copy the tensor's data to a new GPU resource. Comparing to the dataSync() and data(), this method prevents data from being downloaded to CPU.
    For WebGL backend, the data will be stored on a densely packed texture. This means that the texture will use the RGBA channels to store value.
    For WebGPU backend, the data will be stored on a buffer. There is no parameter, so can not use a user-defined size to create the buffer.
    */
    const moduleSetting = loader.getModuleSetting();
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    if (moduleSetting === "tfjs-node-gpu") {
      expect(() => t.dataToGPU()).to.not.throw(Error);
    } else {
      expect(() => t.dataToGPU()).to.throw(Error);
    }
  });
}
