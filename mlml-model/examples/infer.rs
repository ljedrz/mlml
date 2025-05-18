#![recursion_limit = "256"]

use burn::tensor::backend::Backend;
use mlml_model::DeductiveReasoningDataset;

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;

pub fn launch<B: Backend>(device: B::Device, test_vector: Vec<String>) {
    mlml_model::inference::infer::<B, DeductiveReasoningDataset>(
        device,
        "/tmp/mlml_model",
        // Samples from the test dataset, but you are free to test with your own text.
        test_vector,
    );
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use crate::{ElemType, launch};
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    pub fn run(test_vector: Vec<String>) {
        launch::<LibTorch<ElemType>>(LibTorchDevice::Cpu, test_vector);
    }
}

fn main() {
    let f = std::fs::read_to_string("/home/ljedrz/git/ljedrz/mlml/mlml-dataset/test.csv").unwrap();
    let r = f.lines();

    let mut test_vector = vec![];
    for l in r.skip(1) {
        let mut split = l.split(";");
        let expr = split.next().unwrap().to_owned();
        let _res = split.next().unwrap();
        // test_vector.push((expr, res));
        test_vector.push(expr);
    }

    #[cfg(feature = "tch-cpu")]
    tch_cpu::run(test_vector);
}
