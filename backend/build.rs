use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("resnet18-v2-7.onnx")
        .out_dir("model/")
        .run_from_script();
}