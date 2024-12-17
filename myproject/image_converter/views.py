# image_converter/views.py

from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageUploadForm
import os
import torch
import numpy as np
import trimesh
from PIL import Image
from torchvision import transforms

def load_midas_model():
    model_type = "DPT_Large"  # or "MiDaS_small" for faster processing
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    if torch.cuda.is_available():
        midas.to('cuda')
    return midas

def preprocess_image(image_path):
    # Modified preprocessing to ensure consistent dimensions
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Force specific dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    input_batch = transform(image).unsqueeze(0)
    return input_batch

def estimate_depth(model, image_tensor):
    with torch.no_grad():
        prediction = model(image_tensor)
        
        # Ensure output is the correct size
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        
        # Normalize and process the depth map
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        return depth_map

def create_3d_model(depth_map):
    # Normalize depth map
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    h, w = depth_map.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Create vertices
    vertices = np.stack([x.flatten(), y.flatten(), depth_map.flatten() * 100], axis=1)
    
    # Create faces for triangular mesh
    faces = []
    for i in range(h-1):
        for j in range(w-1):
            v0 = i * w + j
            v1 = v0 + 1
            v2 = (i + 1) * w + j
            v3 = v2 + 1
            faces.extend([[v0, v1, v2], [v1, v3, v2]])
    
    faces = np.array(faces)
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def save_3d_model(mesh, output_path):
    mesh.export(output_path)

def convert_to_3d(image_path, output_path):
    try:
        print("Loading model...")
        model = load_midas_model()
        
        print("Preprocessing image...")
        image_tensor = preprocess_image(image_path)
        
        print("Estimating depth...")
        depth_map = estimate_depth(model, image_tensor)
        
        print("Creating 3D model...")
        mesh = create_3d_model(depth_map)
        
        print("Saving 3D model...")
        save_3d_model(mesh, output_path)
        
        print("Conversion complete!")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

def index(request):
    form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = os.path.join('media/uploads', image.name)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            # Save the uploaded file
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Generate output path for the 3D model
            output_path = image_path.replace('.jpg', '.obj').replace('.png', '.obj')
            
            try:
                # Convert the image to 3D
                convert_to_3d(image_path, output_path)

                # Return the 3D model file as a download
                with open(output_path, 'rb') as f:
                    response = HttpResponse(f.read())
                    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(output_path)}"'
                    response['Content-Type'] = 'application/octet-stream'
                return response
            except Exception as e:
                return HttpResponse(f'Error during conversion: {str(e)}', status=500)

    return HttpResponse('Failed to upload file', status=400)


    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = os.path.join('media/uploads', image.name)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            # Save the uploaded file
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            return HttpResponse(f'File uploaded to: {image_path}')
    return HttpResponse('Failed to upload file')
