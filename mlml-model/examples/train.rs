#![recursion_limit = "256"]

use burn::{
    grad_clipping::GradientClippingConfig, nn::transformer::TransformerEncoderConfig,
    optim::AdamWConfig, tensor::backend::AutodiffBackend,
};

use mlml_model::{DeductiveReasoningDataset, training::ExperimentConfig};
use mlml_util::MlmlConfig;

#[cfg(not(any(feature = "f16", feature = "flex32")))]
#[allow(unused)]
type ElemType = f32;

pub fn launch<B: AutodiffBackend>(devices: Vec<B::Device>, mlml_config: MlmlConfig) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(
            mlml_config.model.d_model,
            mlml_config.model.d_ff,
            mlml_config.model.n_heads,
            mlml_config.model.n_layers,
        )
        .with_norm_first(true),
        AdamWConfig::new()
            .with_weight_decay(mlml_config.model.weight_decay)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(
                mlml_config.model.gradient_clipping_norm,
            ))),
    );

    mlml_model::training::train::<B, DeductiveReasoningDataset>(
        devices,
        DeductiveReasoningDataset::train(&mlml_config.dataset.db_path),
        DeductiveReasoningDataset::validate(&mlml_config.dataset.db_path),
        config,
        "/tmp/mlml_model",
        mlml_config,
    );
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };
    use mlml_util::{MlmlConfig, config_path};

    use crate::{ElemType, launch};

    pub fn run() {
        let config_str = std::fs::read_to_string(config_path()).unwrap();
        let config: MlmlConfig = serde_json::from_str(&config_str).unwrap();

        launch::<Autodiff<LibTorch<ElemType>>>(vec![LibTorchDevice::Cpu], config);
    }
}

fn main() {
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
}
