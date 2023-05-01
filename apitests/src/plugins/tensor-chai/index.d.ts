export {};
declare global {
  // type nArray = number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][]};
  namespace Chai {
    interface Assertion {
      haveShape(shape: Array<number>): void;
      haveDtype(dtype: keyof tf.DataTypeMap): void;
      lookLike(
        arr:
          | number[]
          | number[][]
          | number[][][]
          | number[][][][]
          | number[][][][][]
          | number[][][][][][]
      ): void;
    }
  }
}
