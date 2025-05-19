#![recursion_limit = "256"]

use burn::{
    grad_clipping::GradientClippingConfig, nn::transformer::TransformerEncoderConfig,
    optim::AdamWConfig, tensor::backend::AutodiffBackend,
};

use mlml_model::{DeductiveReasoningDataset, training::ExperimentConfig};

#[cfg(not(any(feature = "f16", feature = "flex32")))]
#[allow(unused)]
type ElemType = f32;

pub fn launch<B: AutodiffBackend>(devices: Vec<B::Device>) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 128, 4, 2).with_norm_first(true),
        AdamWConfig::new()
            .with_weight_decay(1e-2)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(3.0))),
    );

    mlml_model::training::train::<B, DeductiveReasoningDataset>(
        devices,
        DeductiveReasoningDataset::train(),
        DeductiveReasoningDataset::validate(),
        config,
        "/tmp/mlml_model",
    );
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<LibTorch<ElemType>>>(vec![LibTorchDevice::Cpu]);
    }
}

fn main() {
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
}
