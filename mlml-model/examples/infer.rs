#![recursion_limit = "256"]

use burn::tensor::backend::Backend;
use mlml_model::RawDataset;
use mlml_util::{MlmlConfig, config_path};

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;

pub fn launch<B: Backend>(
    device: B::Device,
    mlml_config: MlmlConfig,
    test_samples: Vec<(String, String, usize, f32)>,
) {
    mlml_model::inference::infer::<B, RawDataset>(
        device,
        "/tmp/mlml_model",
        test_samples,
        mlml_config,
    );
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use crate::{ElemType, launch};
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use mlml_util::MlmlConfig;

    pub fn run(test_samples: Vec<(String, String, usize, f32)>, mlml_config: MlmlConfig) {
        launch::<LibTorch<ElemType>>(LibTorchDevice::Cpu, mlml_config, test_samples);
    }
}

fn main() {
    let config_str = std::fs::read_to_string(config_path()).unwrap();
    let config: MlmlConfig = serde_json::from_str(&config_str).unwrap();

    let connection = rusqlite::Connection::open(&config.dataset.db_path).unwrap();
    let query = "SELECT * FROM test";
    let mut stmt = connection.prepare(&query).unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut test_samples = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        test_samples.push((
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
            row.get(3).unwrap(),
        ));
    }

    #[cfg(feature = "tch-cpu")]
    tch_cpu::run(test_samples, config);
}
