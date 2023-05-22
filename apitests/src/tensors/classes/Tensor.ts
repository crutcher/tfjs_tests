import * as chai from "chai";
const expect = chai.expect;
import { tensorChaiPlugin } from "../../plugins/tensor-chai";
chai.use(tensorChaiPlugin);
import * as sinon from "sinon";
import type tfTypes from "@tensorflow/tfjs-core";
import * as loader from "../../load-tf";
// utils
import { areEqual } from "../../utils/tensor-utils";

/* MODULE TO LOAD DYNAMICALLY: */
let tf: loader.TFModule;

/* TYPES: */
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
    const moduleSetting = loader.getBackend(tf);
    const validBackends = ["webgl", "webgpu"];

    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    if (!(moduleSetting in validBackends)) {
      expect(() => t.dataToGPU()).to.throw(
        "'readToGPU' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen"
      );
    } else {
      expect(() => t.dataToGPU()).to.not.throw(Error);
    }
  });
  it("  -- Tensor.dataSync()", () => {
    /*
    Synchronously downloads the values from the tf.Tensor.
    This blocks the UI thread until the values are ready, which can cause performance issues.
    */
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    const data: TypedArray = t.dataSync();
    expect(data).to.eql(new Float32Array([2, 3]));
  });
  it("  -- Tensor.dispose()", () => {
    /*
    Synchronously downloads the values from the tf.Tensor.
    This blocks the UI thread until the values are ready, which can cause performance issues.
    */
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    t.dispose();
    expect(() => t.print()).to.throw("Tensor is disposed.");
  });
  it("  -- Tensor.print(verbose?)", () => {
    /*
    Prints the tf.Tensor. See tf.print for details.
    */
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([2, 3]);
    // capture the print output
    const stub = sinon.stub(t, "print").callsFake((verbose = false): string => {
      return t.toString(verbose);
    });
    const output = t.print();
    stub.restore();
    expect(output).to.eql("Tensor\n" + "    [2, 3]");
  });
  it("  -- Tensor.clone()", () => {
    /*
    Returns a copy of the tensor.
    */
    const a: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([
      [1, 2],
      [3, 4],
    ]);
    const b: tfTypes.Tensor<tfTypes.Rank.R1> = a.clone();
    const isEqual: boolean = areEqual(a, b);
    expect(isEqual).to.eql(true);
  });
  it("  -- Tensor.toString()", () => {
    /*
    Returns a human-readable description of the tensor. Useful for logging.
    */
    const t: tfTypes.Tensor<tfTypes.Rank.R1> = tf.tensor([
      [1, 2],
      [3, 4],
    ]);
    const expected = "Tensor\n" + "    [[1, 2],\n" + "     [3, 4]]";
    expect(t.toString()).to.eql(expected);
  });
}
