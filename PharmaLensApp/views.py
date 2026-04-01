from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

def home(request):
    if request.method == "POST" and request.FILES.get("prescription"):
        file = request.FILES["prescription"]

        fs = FileSystemStorage()
        filename = fs.save(file.name, file)

        file_url = fs.url(filename)
        # Here you would typically process the file and extract information
        # For now, we just pass the file URL to the template
        return render(request, "home.html", {
                "file_url": file_url, 
                "uploaded": True
            })
    return render(request, "home.html", {"title": "Home"})