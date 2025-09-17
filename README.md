# AI Image Analysis Application

A Flask-based web service that combines YOLOv9 object detection with Vision Transformer image captioning to provide comprehensive scene understanding and spatial guidance for uploaded images.

## Features

- **Object Detection**: Uses YOLOv9 for accurate object detection with bounding boxes and confidence scores
- **Image Captioning**: Generates natural language descriptions using Vision Transformer (ViT) and GPT-2
- **Spatial Guidance**: Provides positional awareness with left/front/right spatial descriptions
- **Multi-Image Processing**: Handles multiple images in a single request
- **RESTful API**: Simple HTTP endpoint for easy integration

## Technology Stack

- **Backend**: Flask (Python)
- **Object Detection**: YOLOv9 (Ultralytics)
- **Image Captioning**: Vision Encoder-Decoder Model (Transformers)
- **Computer Vision**: OpenCV, PIL
- **Deep Learning**: PyTorch, Torchvision
- **Deployment**: Docker

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Saoudyahya/FLASK-image-guidance-generator-yolo-VisionEncoderDecoderModel.git
   cd FLASK-image-guidance-generator-yolo-VisionEncoderDecoderModel
   ```

2. **Build the Docker image**
   ```bash
   docker build -t image-analysis-app .
   ```

3. **Run the container**
   ```bash
   docker run -p 5001:5001 image-analysis-app
   ```

4. **Test the service**
   ```bash
   curl -X POST -F "images=@your_image.jpg" http://localhost:5001/process_images
   ```

### Local Development

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download YOLOv9 model** (automatic on first run)
   The application will automatically download `yolov9c.pt` when first started.

3. **Run the application**
   ```bash
   python scan.py
   ```

## API Reference

### POST `/process_images`

Process one or multiple images for object detection and captioning.

**Request**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `images` field containing image files

**Response**
```json
[
  {
    "caption": "a person standing in a room with a chair",
    "detected_objects": [
      {
        "label": "person",
        "confidence": 0.95,
        "bounding_box": [100, 150, 300, 450]
      },
      {
        "label": "chair",
        "confidence": 0.87,
        "bounding_box": [400, 200, 550, 400]
      }
    ],
    "guidance": "To your front, there is a person. To your left, there is a chair."
  }
]
```

**Example using curl**
```bash
# Single image
curl -X POST -F "images=@image1.jpg" http://localhost:5001/process_images

# Multiple images
curl -X POST \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  http://localhost:5001/process_images
```

**Example using Python requests**
```python
import requests

url = "http://localhost:5001/process_images"
files = {'images': open('your_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## Application Architecture

### Core Components

1. **ObjectDetection Class**
   - Uses YOLOv9 for real-time object detection
   - Returns detected objects with bounding boxes and confidence scores

2. **ImageCaptioning Class**
   - Employs Vision Transformer with GPT-2 decoder
   - Generates natural language descriptions of scenes

3. **UserGuidance Class**
   - Analyzes object positions relative to camera viewpoint
   - Provides spatial awareness with left/front/right positioning

4. **MainApp Class**
   - Orchestrates the entire processing pipeline
   - Combines detection, captioning, and guidance generation

### Spatial Positioning Logic

The application divides images into three zones for spatial guidance:
- **Left third**: Objects appear to the user's right (camera perspective)
- **Center third**: Objects directly in front
- **Right third**: Objects appear to the user's left (camera perspective)

*Note: Left/right positioning is swapped to match first-person perspective*

## Configuration

### Model Configuration

- **YOLO Model**: `yolov9c.pt` (automatically downloaded)
- **Caption Model**: `nlpconnect/vit-gpt2-image-captioning`
- **Image Processing**: OpenCV with PIL integration

### Server Configuration

- **Host**: `0.0.0.0` (accessible from all interfaces)
- **Port**: `5001`
- **Debug Mode**: Enabled in development

## Dependencies

### Core Libraries
```
flask==3.0.1
torch>=2.2.0
torchvision>=0.17.0
ultralytics
transformers>=4.37.0
opencv-python>=4.9.0.80
pillow>=10.2.0
numpy>=1.26.0
python-dotenv>=1.0.0
werkzeug>=3.0.1
```

### System Requirements
- Python 3.10+
- CUDA-compatible GPU (optional, for faster inference)
- 4GB+ RAM recommended
- Internet connection (for initial model downloads)

## Docker Details

### Base Image
- `python:3.10-slim`
- Includes OpenGL libraries for OpenCV support

### Exposed Ports
- `5001`: Flask application port

### Volume Mounts (Optional)
```bash
# Mount local images directory
docker run -p 5001:5001 -v ./images:/app/images image-analysis-app
```

## Error Handling

The application includes comprehensive error handling:
- Invalid file formats
- Missing image data
- Model loading failures
- Processing errors with detailed error messages

## Performance Considerations

- **First Request**: May be slower due to model loading
- **Subsequent Requests**: Faster processing with loaded models
- **Memory Usage**: Approximately 2-4GB depending on model size
- **GPU Acceleration**: Automatically utilized if CUDA is available

## Use Cases

- **Accessibility Applications**: Scene description for visually impaired users
- **Security Systems**: Automated surveillance with object detection
- **Content Moderation**: Automated image analysis for content filtering
- **Mobile Applications**: Real-time scene understanding for AR/navigation apps
- **Educational Tools**: Interactive learning with image analysis

## Development

### Adding New Features

1. **Custom Object Classes**: Modify YOLO model or add post-processing filters
2. **Enhanced Guidance**: Extend `UserGuidance` class for more detailed descriptions
3. **Multiple Models**: Add model selection via request parameters
4. **Batch Processing**: Implement async processing for large image batches

### Testing

```bash
# Test with sample images
python -m pytest tests/  # Add your test suite
```

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   - Ensure stable internet connection
   - Check disk space for model storage

2. **CUDA Memory Errors**
   - Reduce batch size or use CPU-only mode
   - Monitor GPU memory usage

3. **Image Format Issues**
   - Supported formats: JPG, PNG, BMP, TIFF
   - Ensure images are not corrupted

4. **Port Conflicts**
   - Change port in `app.run()` or Docker configuration
   - Check for other services using port 5001


## Support

For issues and questions:
- Create an issue in the repository
- Check existing documentation
- Review error logs for detailed information
