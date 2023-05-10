import tf from "@tensorflow/tfjs-core";
// Utils
import { TensorUtils } from "../../utils";
import { BasicType } from "../../utils";
import tensorUtils from "../../utils/tensor-utils";

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
    new Assertion(obj.arraySync()).to.eql(arr);
  });

  Assertion.addMethod("filledWith", function filledWith(val: BasicType) {
    const obj: tf.Tensor = utils.flag(this, "object");
    const isFilled = TensorUtils.isFilledWith(val, obj);
    new Assertion(isFilled).to.be.true;
  });
  Assertion.addMethod(
    "allValuesInRange",
    function allValuesInRange(start: number, end: number) {
      const obj: tf.Tensor = utils.flag(this, "object");
      tensorUtils.forEachTensorValue(obj, (val) => {
        new Assertion(val).to.be.within(start, end);
      });
    }
  );
};

// export default tensorChai;
