import tf from "@tensorflow/tfjs-core";

export const tensorChaiPlugin: Chai.ChaiPlugin = function (
  chai: Chai.ChaiStatic,
  utils: Chai.ChaiUtils
) {
  const Assertion = chai.Assertion;

  // your helpers here
  Assertion.addMethod("haveShape", function haveShape(arr) {
    const obj: tf.Tensor = utils.flag(this, "object");
    new Assertion(obj.shape).to.eql(arr);
  });
};

// export default tensorChai;
