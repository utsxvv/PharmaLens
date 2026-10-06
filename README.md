## 📌 About

**PharmaLens** is a web-based Deep Learning application designed to extract text from handwritten and printed pharmaceutical labels and prescriptions using a custom **CRNN (CNN + BiLSTM + CTC Loss)** model, automatically mapping recognized medicines to their corresponding **Generic Drug Names**.

Built to streamline medicine identification, this system focuses on **high OCR accuracy, real-time image quality validation, and fuzzy search post-processing** for practical healthcare applications.

No clutter — just an intelligent OCR system tailored for real-world pharmaceutical workflows.

## ⚙️ Tech Stack

- **Machine Learning**: PyTorch, Torchvision
- **Backend**: Python, FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing & Matching**: OpenCV, RapidFuzz, Pandas, NumPy

## ✨ Features

- 🔍 Extract text from handwritten and printed medicine labels  
- 🧠 CRNN architecture (CNN + BiLSTM + CTC Loss) for sequence recognition  
- 🎯 Automated **Generic Drug Name** mapping using **RapidFuzz** matching  
- 📷 Real-time image quality checks (Blur detection, Illumination & Contrast validation)  
- ⚙️ Dynamic image enhancement (CLAHE contrast boosting & Denoising)  
- 🌐 Clean web interface with drag-and-drop image upload  
- 🚀 Fast REST API endpoint (`/predict`) for programmatic integration  

## 📜 Acknowledgement

This project was developed collaboratively with my teammate, [Shreyansh Padhiyar](https://github.com/ShreyanshPadhiyar10).

## 📬 Connect

For questions or support, contact:

- **Gmail**: [utssavvpatel@gmail.com](mailto:utssavvpatel@gmail.com)
- **LinkedIn**: [Utsav Patel](https://www.linkedin.com/in/utsxvv)
- **GitHub**: [@utsxvv](https://github.com/utsxvv)

<p align="center">
  Made with ❤️ by UTSAV
</p>