import { BasicType, TFArray, BoolArray } from "../../utils";
export {};
declare global {
  // type nArray = number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][]};
  namespace Chai {
    interface Assertion {
      haveShape(shape: Array<number>): void;
      haveSize(size: number): void;
      haveDtype(dtype: keyof tf.DataTypeMap): void;
      lookLike(arr: TFArray | BoolArray): void;
      filledWith(val: BasicType): void;
      allOnes: Assertion;
      allZeros: Assertion;
      allValuesInRange(start: number, end: number): void;
      onlyValuesInSet(arr: number[]): void;
    }
  }
}
