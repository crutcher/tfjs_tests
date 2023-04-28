export {};
declare global {
  namespace Chai {
    interface Assertion {
      haveShape(shape: Array<number>): void;
    }
  }
}
