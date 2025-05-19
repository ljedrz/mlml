#![recursion_limit = "256"]

use burn::tensor::backend::Backend;
use mlml_model::DeductiveReasoningDataset;

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;

pub fn launch<B: Backend>(device: B::Device, test_samples: Vec<(String, String)>) {
    mlml_model::inference::infer::<B, DeductiveReasoningDataset>(
        device,
        "/tmp/mlml_model",
        test_samples,
    );
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use crate::{ElemType, launch};
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    pub fn run(test_samples: Vec<(String, String)>) {
        launch::<LibTorch<ElemType>>(LibTorchDevice::Cpu, test_samples);
    }
}

fn main() {
    let connection =
        rusqlite::Connection::open("/home/ljedrz/git/ljedrz/mlml/mlml-dataset/dataset.db").unwrap();
    let query = "SELECT * FROM test";
    let mut stmt = connection.prepare(&query).unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut test_samples = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        test_samples.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }

    #[cfg(feature = "tch-cpu")]
    tch_cpu::run(test_samples);
}
