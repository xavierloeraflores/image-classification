mod model;
use axum::{
    extract::Multipart,
    routing::get,
    routing::post,
    response::Json,
    Router
};
use std::net::SocketAddr;
use image::ImageReader;
use std::io::Cursor;
use burn_ndarray::{NdArray, NdArrayDevice};
use model::my_model::Model;
use burn::tensor::{backend::Backend, Device, Element, Tensor, TensorData};

const HEIGHT: usize = 224;
const WIDTH: usize = 224;

#[tokio::main]
async fn main() {
    // build our application with a single route
    let app = Router::new().route("/", get(root))
    .route("/image", post(classify_image));

    // Get the port number from the environment, default to 3000
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "3000".to_string()) // Get the port as a string or default to "3000"
        .parse() // Parse the port string into a u16
        .expect("Failed to parse PORT");

    // Create a socket address (IPv6 binding)
    let address = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(&address).await.unwrap();

    println!("Listening on PORT: {}", address.port());

    // Run the app with hyper, listening on the specified address
    axum::serve(listener, app).await.unwrap();
}

// basic handler that responds with a static string
async fn root() -> &'static str {
    "Hello World!"
}

async fn classify_image(mut multipart: Multipart) -> Json<bool>{

    while let Some(field) = multipart.next_field().await.unwrap(){
        let name = field.name().unwrap().to_string();
        let data = field.bytes().await.unwrap();
        println!("File Accepted: '{}'", name);
        println!("Data Size: {:?}", data.len());
        if name=="image"{
            let processed_image_result = preprocess_image(data.to_vec());
            match processed_image_result{
                Ok(processed_image)=>{
                    model_run_image(processed_image);
                    return Json(true)
                }
                Err(e)=>println!("Error {e}")
            }
        }
    }
    println!("Finished");

    Json(true)
}

fn preprocess_image(data: Vec<u8>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let img = ImageReader::new(Cursor::new(data))
        .with_guessed_format()?
        .decode()?;
    let resized_img = img.resize_exact(224, 224, image::imageops::FilterType::Nearest);

    let rgb_img = resized_img.to_rgb8();
    let input_data: Vec<f32> = rgb_img
    .pixels()
        .flat_map(|p| {
            p.0.iter().map(|&channel| (channel as f32) / 255.0).collect::<Vec<f32>>()
        })
        .collect();

    assert_eq!(input_data.len(), WIDTH * HEIGHT * 3);

    Ok(input_data)
}

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(data, shape).convert::<B::FloatElem>(), device)
        .permute([2, 0, 1]) // [C, H, W]
        / 255 // normalize between [0, 1]
}



fn model_run_image(processed_image:Vec<f32>) {

    let device = NdArrayDevice::default();
    let img_tensor = to_tensor(
        processed_image,
        [HEIGHT, WIDTH, 3],
        &device,
    )
    .unsqueeze::<4>(); // [B, C, H, W]
    let model: Model<NdArray<f32>>= Model::default();
    // let input= tensor::Tensor::from_data(data_in, &device);
    let output = model.forward(img_tensor);
    println!("{:?}",output);
    let (score, idx) = output.max_dim_with_indices(1);
    let idx = idx.into_scalar() as usize;
    println!(
        "Predicted: XX\nCategory Id: {}\nScore: {:.4}",
        idx,
        score.into_scalar()
    );
}