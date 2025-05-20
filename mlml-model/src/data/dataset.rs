// Each dataset struct has a field for the underlying SQLite dataset and implements methods for
// accessing and processing the data. Each dataset is also provided with specific information about
// its classes via the MlmlDataset trait. These implementations are designed to be used with a
// machine learning framework for tasks such as training a text classification model.

use std::path::Path;

use burn::data::dataset::{Dataset, SqliteDataset, SqliteDatasetStorage};

// Define a struct for text classification items
#[derive(new, Clone, Debug)]
pub struct MlmlItem {
    pub text: String, // The text for classification
    pub label: usize, // The label of the text (classification category)
}

// Trait for text classification datasets
pub trait MlmlDataset: Dataset<MlmlItem> {
    fn class_name(label: &str) -> String; // Returns the name of the class given its label
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RawItem {
    pub expression: String, // The text for classification
    pub result: String,     // The label of the text (classification category)
}

pub struct RawDataset {
    dataset: SqliteDataset<RawItem>,
}

impl Dataset<MlmlItem> for RawDataset {
    fn get(&self, index: usize) -> Option<MlmlItem> {
        self.dataset
            .get(index)
            .map(|item| MlmlItem::new(item.expression, (item.result == "true") as usize))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl RawDataset {
    pub fn new(db_path: &Path, split: &str) -> Self {
        let dataset: SqliteDataset<RawItem> = SqliteDatasetStorage::from_file(db_path)
            .reader(split)
            .unwrap();
        Self { dataset }
    }

    pub fn train(db_path: &Path) -> Self {
        Self::new(db_path, "train")
    }

    pub fn validate(db_path: &Path) -> Self {
        Self::new(db_path, "valid")
    }
}

impl MlmlDataset for RawDataset {
    fn class_name(label: &str) -> String {
        label.to_owned()
    }
}
