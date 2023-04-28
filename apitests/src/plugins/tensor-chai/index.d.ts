export {};
declare global {
  namespace Chai {
    interface Assertion {
      haveShape(shape: Array<number>): void;
      haveDtype(dtype: keyof tf.DataTypeMap): void;
    }
  }
}
