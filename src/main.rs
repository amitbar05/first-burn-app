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

type GpuTrainBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;
type GpuInferBackend = burn::backend::wgpu::Wgpu;
type CpuTrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
type CpuInferBackend = burn::backend::NdArray<f32>;

// 6-layer MLP: 784 → 4096 → 4096 → 2048 → 1024 → 256 → 10
// ~30M parameters — enough compute to saturate a GPU
#[derive(Module, Debug)]
pub struct MnistModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    fc4: Linear<B>,
    fc5: Linear<B>,
    fc6: Linear<B>,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct MnistModelConfig {
    #[config(default = "784")]
    input_size: usize,
    #[config(default = "10")]
    output_size: usize,
}

impl MnistModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MnistModel<B> {
        MnistModel {
            fc1: LinearConfig::new(self.input_size, 4096).init(device),
            fc2: LinearConfig::new(4096, 4096).init(device),
            fc3: LinearConfig::new(4096, 2048).init(device),
            fc4: LinearConfig::new(2048, 1024).init(device),
            fc5: LinearConfig::new(1024, 256).init(device),
            fc6: LinearConfig::new(256, self.output_size).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> MnistModel<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.fc1.forward(x));
        let x = self.relu.forward(self.fc2.forward(x));
        let x = self.relu.forward(self.fc3.forward(x));
        let x = self.relu.forward(self.fc4.forward(x));
        let x = self.relu.forward(self.fc5.forward(x));
        self.fc6.forward(x)
    }
}

const EPOCHS: usize = 10;
const BATCH_SIZE: usize = 1024;
const LR: f64 = 0.001;

fn main() {
    println!("=== MNIST Benchmark: CPU vs GPU (same config) ===");
    println!("Model: 784→4096→4096→2048→1024→256→10 (~30M params)");
    println!("Config: epochs={}, batch_size={}, lr={}\n", EPOCHS, BATCH_SIZE, LR);

    let train_data = burn_dataset::vision::MnistDataset::train();
    let test_data = burn_dataset::vision::MnistDataset::test();
    println!("Train: {} samples, Test: {} samples\n", train_data.len(), test_data.len());

    // --- CPU run ---
    println!("========== CPU (NdArray) ==========");
    let cpu_device: burn::backend::ndarray::NdArrayDevice = Default::default();
    let cpu_result = train_and_eval::<CpuTrainBackend, CpuInferBackend>(
        &train_data, &test_data, &cpu_device, "CPU",
    );

    // --- GPU run ---
    println!("\n========== GPU (WGPU) ==========");
    let gpu_device = burn::backend::wgpu::WgpuDevice::default();
    let gpu_result = train_and_eval::<GpuTrainBackend, GpuInferBackend>(
        &train_data, &test_data, &gpu_device, "GPU",
    );

    // --- Summary ---
    println!("\n========== SUMMARY ==========");
    println!(
        "CPU: {:.2}s total, {:.2}s/epoch, final acc {:.2}%",
        cpu_result.total_time,
        cpu_result.total_time / EPOCHS as f64,
        cpu_result.epoch_accs.last().unwrap()
    );
    println!(
        "GPU: {:.2}s total, {:.2}s/epoch, final acc {:.2}%",
        gpu_result.total_time,
        gpu_result.total_time / EPOCHS as f64,
        gpu_result.epoch_accs.last().unwrap()
    );
    println!(
        "Speedup: {:.1}x",
        cpu_result.total_time / gpu_result.total_time
    );

    // --- Plot both ---
    save_loss_plot(&cpu_result, &gpu_result);
    println!("\nLoss plot saved to loss_plot.png");
}

struct RunResult {
    epoch_losses: Vec<f32>,
    epoch_accs: Vec<f32>,
    #[allow(dead_code)]
    epoch_times: Vec<f64>,
    total_time: f64,
}

fn train_and_eval<TrainB, InferB>(
    train_data: &burn_dataset::vision::MnistDataset,
    test_data: &burn_dataset::vision::MnistDataset,
    device: &TrainB::Device,
    label: &str,
) -> RunResult
where
    TrainB: burn::tensor::backend::AutodiffBackend<InnerBackend = InferB>,
    InferB: Backend,
{
    let t0 = Instant::now();
    let (train_images, train_labels) = load_dataset::<TrainB>(train_data, device);
    let (test_images, test_labels) = load_dataset::<TrainB>(test_data, device);
    println!("{} data loaded in {:.2?}", label, t0.elapsed());

    let model_config = MnistModelConfig::new();
    let mut model: MnistModel<TrainB> = model_config.init(device);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = CrossEntropyLoss::new(None, device);

    let num_train = train_data.len();
    let num_batches = (num_train + BATCH_SIZE - 1) / BATCH_SIZE;

    let mut result = RunResult {
        epoch_losses: Vec::new(),
        epoch_accs: Vec::new(),
        epoch_times: Vec::new(),
        total_time: 0.0,
    };

    let train_start = Instant::now();

    for epoch in 0..EPOCHS {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0f32;

        for batch_idx in 0..num_batches {
            let start = batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(num_train);

            let images = train_images.clone().slice([start..end, 0..784]);
            let labels = train_labels.clone().slice([start..end]);

            let output = model.forward(images);
            let loss = loss_fn.forward(output, labels);

            let loss_val: f32 = loss.to_data().to_vec().unwrap()[0];
            total_loss += loss_val;

            let grads = loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(LR, model, grad_params);
        }

        let valid_model = model.valid();
        let (correct, total) = evaluate::<InferB>(
            &valid_model,
            &test_images.clone().inner(),
            &test_labels.clone().inner(),
        );

        let avg_loss = total_loss / num_batches as f32;
        let acc = correct as f32 / total as f32 * 100.0;
        let elapsed = epoch_start.elapsed().as_secs_f64();

        result.epoch_losses.push(avg_loss);
        result.epoch_accs.push(acc);
        result.epoch_times.push(elapsed);

        println!(
            "  {} Epoch {:2}/{} -- Loss: {:.4} -- Acc: {:.2}% -- {:.2}s",
            label, epoch + 1, EPOCHS, avg_loss, acc, elapsed
        );
    }

    result.total_time = train_start.elapsed().as_secs_f64();
    println!(
        "\n{} done: {:.2}s total, final acc {:.2}%\n",
        label,
        result.total_time,
        result.epoch_accs.last().unwrap()
    );

    result
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

fn evaluate<B: Backend>(
    model: &MnistModel<B>,
    images: &Tensor<B, 2>,
    labels: &Tensor<B, 1, Int>,
) -> (usize, usize) {
    let n = labels.dims()[0];
    let num_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;
    let mut correct = 0usize;
    let mut total = 0usize;

    for batch_idx in 0..num_batches {
        let start = batch_idx * BATCH_SIZE;
        let end = (start + BATCH_SIZE).min(n);

        let batch_images = images.clone().slice([start..end, 0..784]);
        let batch_labels = labels.clone().slice([start..end]);

        let output = model.forward(batch_images);
        let preds = output.argmax(1).reshape([end - start]);
        let matches = preds.equal(batch_labels);
        let batch_correct: f32 = matches.float().sum().to_data().to_vec().unwrap()[0];
        correct += batch_correct as usize;
        total += end - start;
    }

    (correct, total)
}

fn save_loss_plot(cpu: &RunResult, gpu: &RunResult) {
    use plotters::prelude::*;

    let root = BitMapBackend::new("loss_plot.png", (1000, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let all_losses: Vec<f32> = cpu.epoch_losses.iter().chain(gpu.epoch_losses.iter()).cloned().collect();
    let max_loss = all_losses.iter().cloned().fold(0.0f32, f32::max);
    let min_loss = all_losses.iter().cloned().fold(f32::MAX, f32::min);
    let range = if max_loss - min_loss > 0.0 { max_loss - min_loss } else { 1.0 };

    let n = cpu.epoch_losses.len().max(gpu.epoch_losses.len());

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "MNIST Loss — CPU vs GPU (6-layer MLP, batch={}, lr={}, epochs={})",
                BATCH_SIZE, LR, EPOCHS
            ),
            ("sans-serif", 22).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(1..n + 1, (min_loss - range * 0.1)..(max_loss + range * 0.1))
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .x_labels(n)
        .y_labels(8)
        .draw()
        .unwrap();

    // CPU line (blue)
    let cpu_points: Vec<(usize, f32)> = cpu.epoch_losses.iter().enumerate().map(|(i, &l)| (i + 1, l)).collect();
    chart
        .draw_series(LineSeries::new(cpu_points.clone(), &BLUE))
        .unwrap()
        .label(format!("CPU (NdArray) — {:.1}s", cpu.total_time))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    chart
        .draw_series(cpu_points.iter().map(|&(x, y)| Circle::new((x, y), 4, BLUE.filled())))
        .unwrap();

    // GPU line (red)
    let gpu_points: Vec<(usize, f32)> = gpu.epoch_losses.iter().enumerate().map(|(i, &l)| (i + 1, l)).collect();
    chart
        .draw_series(LineSeries::new(gpu_points.clone(), &RED))
        .unwrap()
        .label(format!("GPU (WGPU) — {:.1}s", gpu.total_time))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    chart
        .draw_series(gpu_points.iter().map(|&(x, y)| Circle::new((x, y), 4, RED.filled())))
        .unwrap();

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();

    root.present().unwrap();
}
