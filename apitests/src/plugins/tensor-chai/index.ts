import tf from "@tensorflow/tfjs-core";
// Utils
import { TensorUtils } from "../../utils";
import { BasicType } from "../../utils";

export const tensorChaiPlugin: Chai.ChaiPlugin = function (
  chai: Chai.ChaiStatic,
  utils: Chai.ChaiUtils
) {
  const Assertion = chai.Assertion;

  // new chai assertions : " expect(tensor).to.haveShape "
  Assertion.addMethod("haveShape", function haveShape(arr) {
    const obj: tf.Tensor = utils.flag(this, "object");
    new Assertion(obj.shape).to.eql(arr);
  });

  Assertion.addMethod("haveSize", function haveSize(num: number) {
    const obj: tf.Tensor = utils.flag(this, "object");
    new Assertion(obj.size).to.eql(num);
  });

  Assertion.addMethod(
    "haveDtype",
    function haveDtype(dtype: keyof tf.DataTypeMap) {
      const obj: tf.Tensor = utils.flag(this, "object");
      new Assertion(obj.dtype).to.eql(dtype);
    }
  );

  Assertion.addMethod("lookLike", function lookLike(arr) {
    const obj: tf.Tensor = utils.flag(this, "object");
    if (obj.dtype === "bool") {
      new Assertion(TensorUtils.asBoolArray(obj)).to.eql(arr);
    } else {
      new Assertion(obj.arraySync()).to.eql(arr);
    }
  });

  Assertion.addMethod("filledWith", function filledWith(val: BasicType) {
    const obj: tf.Tensor = utils.flag(this, "object");
    const isFilled = TensorUtils.isTrueForEach(obj, (item) => item === val);
    new Assertion(isFilled).to.be.true;
  });

  Assertion.addProperty("allZeros", function allZerosTest() {
    const obj: tf.Tensor = utils.flag(this, "object");
    const isAllZeros = TensorUtils.isAllZeros(obj);
    new Assertion(isAllZeros).to.be.true;
  });

  Assertion.addProperty("allOnes", function allZerosTest() {
    const obj: tf.Tensor = utils.flag(this, "object");
    const isAllOnes = TensorUtils.isAllOnes(obj);
    new Assertion(isAllOnes).to.be.true;
  });

  Assertion.addMethod(
    "allValuesInRange",
    function allValuesInRange(start: number, end: number) {
      const obj: tf.Tensor = utils.flag(this, "object");
      TensorUtils.forEachTensorValue(obj, (val) => {
        new Assertion(val).to.be.within(start, end);
      });
    }
  );

  Assertion.addMethod(
    "onlyValuesInSet",
    function onlyValuesInSet(arr: number[]) {
      const obj: tf.Tensor = utils.flag(this, "object");
      TensorUtils.forEachTensorValue(obj, (val) => {
        new Assertion(arr).to.include(val);
      });
    }
  );
};
