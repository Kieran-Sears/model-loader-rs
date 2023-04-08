use anyhow::Result;
use image::{self, imageops::FilterType};
use serde::Serialize;
use tensorflow::{
    Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
};

#[derive(Debug)]
pub struct ModelInput(Vec<f32>);

impl ModelInput {
    pub fn from_image_bytes(bytes: Vec<u8>, height: u64, width: u64) -> Result<Self> {
        const NORM_SCALE: f32 = 1. / 255.;
        let im = image::load_from_memory(&bytes)?
            .resize_exact(height as u32, width as u32, FilterType::Nearest)
            .grayscale()
            .as_bytes()
            .to_vec()
            .into_iter()
            .map(|x| (x as f32) * NORM_SCALE)
            .collect::<Vec<f32>>();
        Ok(Self(im))
    }
}

#[derive(Debug, Serialize)]
pub struct ModelPrediction {
    pub label: u8,
    pub confidence: f32,
}

pub struct Model {
    bundle: SavedModelBundle,
    input_op: Operation,
    input_index: i32,
    output_op: Operation,
    output_index: i32,
}

impl Model {
    pub fn from_dir<P: AsRef<str>>(
        model_dir: P,
        signature_key: &str,
        input_key: &str,
        output_key: &str,
    ) -> Result<Self> {
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(),
            &[DEFAULT_SERVING_SIGNATURE_DEF_KEY],
            &mut graph,
            model_dir.as_ref(),
        )?;

        let sig = bundle.meta_graph_def().get_signature(signature_key)?;
        let input_info = sig.get_input(input_key)?;
        let output_info = sig.get_output(output_key)?;
        let input_op = graph.operation_by_name_required(&input_info.name().name)?;
        let output_op = graph.operation_by_name_required(&output_info.name().name)?;
        let input_index = input_info.name().index;
        let output_index = output_info.name().index;

        Ok(Self {
            bundle,
            input_op,
            input_index,
            output_op,
            output_index,
        })
    }

    pub fn predict(&self, input: ModelInput) -> Result<ModelPrediction> {
        let input_dims = &[1, input.0.len() as u64];
        let input_tensor = Tensor::<f32>::new(input_dims).with_values(&input.0)?;
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&self.input_op, self.input_index, &input_tensor);
        let output_fetch = run_args.request_fetch(&self.output_op, self.output_index);
        self.bundle.session.run(&mut run_args)?;

        let output = run_args.fetch::<f32>(output_fetch)?;
        let mut confidence = 0f32;
        let mut label = 0u8;
        for i in 0..output.dims()[1] {
            let conf = output[i as usize];
            if conf > confidence {
                confidence = conf;
                label = i as u8;
            }
        }

        Ok(ModelPrediction { label, confidence })
    }
}