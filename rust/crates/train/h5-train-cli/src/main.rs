//! Training interface layer. The GPU trainer lives in Python/PyTorch on Colab.
//! This Rust CLI only exposes pure metrics for offline scoring of predictions.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] != "--metrics-demo" {
        println!("h5-train-cli --metrics-demo");
        println!("GPU training runs in bridge/training (PyTorch) on Colab.");
        return;
    }
    let pred = [2.0, 3.0, 4.0];
    let truth = [2.0, 4.0, 4.0];
    println!("MAE={:.4}", h5_train::mae(&pred, &truth));
    println!("MSE={:.4}", h5_train::mse(&pred, &truth));
}
