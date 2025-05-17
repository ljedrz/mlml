// The DeductiveReasoningDataset and DbPediaDataset structs are examples of specific text
// classification datasets.  Each dataset struct has a field for the underlying
// SQLite dataset and implements methods for accessing and processing the data.
// Each dataset is also provided with specific information about its classes via
// the TextClassificationDataset trait. These implementations are designed to be used
// with a machine learning framework for tasks such as training a text classification model.

use burn::data::dataset::{Dataset, SqliteDataset, SqliteDatasetStorage};

// Define a struct for text classification items
#[derive(new, Clone, Debug)]
pub struct TextClassificationItem {
    pub text: String, // The text for classification
    pub label: usize, // The label of the text (classification category)
}

// Trait for text classification datasets
pub trait TextClassificationDataset: Dataset<TextClassificationItem> {
    fn class_name(label: &str) -> String; // Returns the name of the class given its label
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeductiveReasoningItem {
    pub expression: String, // The text for classification
    pub result: String,     // The label of the text (classification category)
}

pub struct DeductiveReasoningDataset {
    dataset: SqliteDataset<DeductiveReasoningItem>, // Underlying SQLite dataset
}

impl Dataset<TextClassificationItem> for DeductiveReasoningDataset {
    /// Returns a specific item from the dataset
    fn get(&self, index: usize) -> Option<TextClassificationItem> {
        self.dataset.get(index).map(|item| {
            TextClassificationItem::new(item.expression, (item.result == "True") as usize)
        }) // Map DeductiveReasoningItems to TextClassificationItems
    }

    /// Returns the length of the dataset
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DeductiveReasoningDataset {
    /// Returns the training portion of the dataset
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Returns the testing portion of the dataset
    pub fn test() -> Self {
        Self::new("test")
    }

    /// Constructs the dataset from a split (either "train" or "test")
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<DeductiveReasoningItem> =
            SqliteDatasetStorage::from_file("/home/ljedrz/git/ljedrz/mlml/mlml-dataset/dataset.db")
                .reader(split)
                .unwrap();
        Self { dataset }
    }
}

impl TextClassificationDataset for DeductiveReasoningDataset {
    /// Returns the name of a class given its label
    fn class_name(label: &str) -> String {
        label.to_owned()
    }
}
