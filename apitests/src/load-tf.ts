export default async function () {
  if (process.env.OS === "linux") {
    return await import("@tensorflow/tfjs-node-gpu");
  } else {
    return await import("@tensorflow/tfjs");
  }
}
