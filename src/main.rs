mod model_loader;

use std::path::PathBuf;
use model_loader::{Model, ModelInput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new Model instance from the directory containing the saved model.
    let model_dir = PathBuf::from("path/to/model/dir");
    let model_dir_str = model_dir.to_str().unwrap();
    let model = Model::from_dir(model_dir_str, "serving_default", "input_1", "dense_2/Softmax").unwrap();


    // Create a ModelInput instance from a vector of image bytes.
    let image_bytes = vec![0u8; 10000]; // hypothetical image bytes
    let input = ModelInput::from_image_bytes(image_bytes, 28, 28)?;

    // Get a prediction from the model using the input.
    let prediction = model.predict(input)?;

    // Print the predicted label and confidence.
    println!("Predicted label: {}", prediction.label);
    println!("Confidence: {}", prediction.confidence);

    Ok(())
}
