from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .utils.preprocessing import preprocess_image
import cv2
import os

def home(request):
    if request.method == "POST" and request.FILES.get("prescription"):
        file = request.FILES["prescription"]

        fs = FileSystemStorage()
        filename = fs.save(file.name, file)

        # full file path (important for OpenCV)
        filepath = fs.path(filename)

        # ✅ STEP 1: preprocess image
        processed = preprocess_image(filepath)

        # ✅ STEP 2: save processed image
        name, ext = os.path.splitext(filename)
        processed_filename = name + "_processed" + ext
        processed_path = fs.path(processed_filename)

        cv2.imwrite(processed_path, processed)

        # URLs for frontend
        file_url = fs.url(filename)
        processed_url = fs.url(processed_filename)

        return render(request, "home.html", {
            "file_url": file_url,
            "processed_url": processed_url,
            "uploaded": True
        })

    return render(request, "home.html", {"title": "Home"})