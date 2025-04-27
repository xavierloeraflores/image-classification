use axum::{
    extract::Multipart,
    routing::get,
    routing::post,
    response::Json,
    Router
};
use std::net::SocketAddr;
use image::{GenericImageView, ImageReader};
use std::io::{Cursor, Bytes};

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
    }

    Json(true)
}

fn preprocess_image(data: Vec<u8>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let img = ImageReader::new(Cursor::new(data))
        .with_guessed_format()?
        .decode()?;
    let resized_img = img.resize(224, 224, image::imageops::FilterType::Nearest);

    let rgb_img = resized_img.to_rgb8();

    let input_data: Vec<f32> = rgb_img
    .pixels()
        .flat_map(|p| {
            p.0.iter().map(|&channel| (channel as f32) / 255.0).collect::<Vec<f32>>()
        })
        .collect();

    Ok(input_data)
}