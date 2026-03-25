#![recursion_limit = "512"]

use burn::config::Config;
use burn::data::dataset::Dataset;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Tensor;
use std::time::Instant;

type TrainBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;
type InferBackend = burn::backend::wgpu::Wgpu;

#[derive(Module, Debug)]
pub struct MnistModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    relu1: Relu,
    relu2: Relu,
}

#[derive(Config, Debug)]
pub struct MnistModelConfig {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

impl MnistModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MnistModel<B> {
        MnistModel {
            fc1: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            fc2: LinearConfig::new(self.hidden_size, self.hidden_size / 2).init(device),
            fc3: LinearConfig::new(self.hidden_size / 2, self.output_size).init(device),
            relu1: Relu::new(),
            relu2: Relu::new(),
        }
    }
}

impl<B: Backend> MnistModel<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.relu1.forward(self.fc1.forward(x));
        let x = self.relu2.forward(self.fc2.forward(x));
        self.fc3.forward(x)
    }
}

fn main() {
    let epochs = 5;
    let batch_size = 256;
    let lr = 0.001;

    println!("Loading MNIST dataset...");
    let t0 = Instant::now();
    let train_data = burn_dataset::vision::MnistDataset::train();
    let test_data = burn_dataset::vision::MnistDataset::test();
    println!(
        "Train: {} samples, Test: {} samples (loaded in {:.2?})",
        train_data.len(),
        test_data.len(),
        t0.elapsed()
    );

    let device = burn::backend::wgpu::WgpuDevice::default();

    println!("Transferring data to GPU...");
    let t1 = Instant::now();
    let (train_images, train_labels) = load_dataset::<TrainBackend>(&train_data, &device);
    let (test_images, test_labels) = load_dataset::<TrainBackend>(&test_data, &device);
    println!("Data on GPU in {:.2?}", t1.elapsed());

    let model_config = MnistModelConfig::new(784, 128, 10);
    let mut model: MnistModel<TrainBackend> = model_config.init(&device);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = CrossEntropyLoss::new(None, &device);

    let num_train = train_data.len();
    let num_batches = (num_train + batch_size - 1) / batch_size;

    println!(
        "\nTraining: epochs={}, batch={}, lr={}, batches/epoch={}\n",
        epochs, batch_size, lr, num_batches
    );

    let mut epoch_losses: Vec<f32> = Vec::new();
    let train_start = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0f32;
        let mut loss_samples = 0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(num_train);

            let images = train_images.clone().slice([start..end, 0..784]);
            let labels = train_labels.clone().slice([start..end]);

            let output = model.forward(images);
            let loss = loss_fn.forward(output, labels);

            // Only sync loss to CPU for periodic logging
            if batch_idx % 200 == 0 || batch_idx == num_batches - 1 {
                let loss_val: f32 = loss.to_data().to_vec().unwrap()[0];
                total_loss += loss_val;
                loss_samples += 1;
                if batch_idx % 200 == 0 {
                    println!(
                        "  Epoch {} Batch {}/{} Loss: {:.4}",
                        epoch + 1,
                        batch_idx,
                        num_batches,
                        loss_val
                    );
                }
            }

            let grads = loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grad_params);
        }

        // Evaluate using inner backend (no autodiff overhead)
        let valid_model = model.valid();
        let (correct, total) = evaluate(
            &valid_model,
            &test_images.clone().inner(),
            &test_labels.clone().inner(),
            batch_size,
        );

        let avg_loss = if loss_samples > 0 {
            total_loss / loss_samples as f32
        } else {
            0.0
        };
        epoch_losses.push(avg_loss);
        let acc = correct as f32 / total as f32 * 100.0;
        println!(
            "Epoch {} -- Loss: {:.4} -- Acc: {:.2}% -- {:.2?}\n",
            epoch + 1,
            avg_loss,
            acc,
            epoch_start.elapsed()
        );
    }

    println!("Training complete in {:.2?}", train_start.elapsed());
    println!("Loss per epoch: {:?}", epoch_losses);

    save_loss_plot(&epoch_losses);
    println!("Loss plot saved to loss_plot.png");
}

fn load_dataset<B: Backend>(
    dataset: &burn_dataset::vision::MnistDataset,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
    let n = dataset.len();
    let mut flat_images = Vec::with_capacity(n * 784);
    let mut flat_labels = Vec::with_capacity(n);

    for item in dataset.iter() {
        for row in &item.image {
            for &pixel in row {
                flat_images.push(pixel / 255.0);
            }
        }
        flat_labels.push(item.label as i64);
    }

    let images: Tensor<B, 1> = Tensor::from_data(&flat_images[..], device);
    let images: Tensor<B, 2> = images.reshape([n, 784]);
    let labels: Tensor<B, 1, Int> = Tensor::from_data(&flat_labels[..], device);

    (images, labels)
}

fn evaluate(
    model: &MnistModel<InferBackend>,
    images: &Tensor<InferBackend, 2>,
    labels: &Tensor<InferBackend, 1, Int>,
    batch_size: usize,
) -> (usize, usize) {
    let n = labels.dims()[0];
    let num_batches = (n + batch_size - 1) / batch_size;
    let mut correct = 0usize;
    let mut total = 0usize;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(n);

        let batch_images = images.clone().slice([start..end, 0..784]);
        let batch_labels = labels.clone().slice([start..end]);

        let output = model.forward(batch_images);
        let preds = output.argmax(1).reshape([end - start]);
        let matches = preds.equal(batch_labels);
        let batch_correct: i32 = matches.int().sum().to_data().to_vec().unwrap()[0];
        correct += batch_correct as usize;
        total += end - start;
    }

    (correct, total)
}

fn save_loss_plot(epoch_losses: &[f32]) {
    use plotters::prelude::*;

    let root = BitMapBackend::new("loss_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_loss = epoch_losses.iter().cloned().fold(0.0f32, f32::max);
    let min_loss = epoch_losses.iter().cloned().fold(f32::MAX, f32::min);
    let loss_range = if max_loss - min_loss > 0.0 {
        max_loss - min_loss
    } else {
        1.0
    };

    let mut chart = ChartBuilder::on(&root)
        .caption("MNIST Training Loss", ("sans-serif", 30).into_font())
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0..epoch_losses.len(),
            (min_loss - loss_range * 0.1)..(max_loss + loss_range * 0.1),
        )
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(epoch_losses.len())
        .y_labels(5)
        .draw()
        .unwrap();

    let points: Vec<(usize, f32)> = epoch_losses
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l))
        .collect();

    chart
        .draw_series(LineSeries::new(points, &RED))
        .unwrap();

    chart
        .draw_series(
            epoch_losses
                .iter()
                .enumerate()
                .map(|(i, &l)| Circle::new((i, l), 5, RED.filled())),
        )
        .unwrap();

    root.present().unwrap();
}
