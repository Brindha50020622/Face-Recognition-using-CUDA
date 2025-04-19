# Face-Recognition-using-CUDA


Dataset: AT&T Database of Faces

The project implements a facial recognition system using Local Binary Patterns (LBP) for 
feature extraction and CUDA-based parallel computing for performance optimization. The 
system creates LBP histograms from facial images, compares them using chi-square distance, 
and identifies individuals based on the nearest match. 
The facial recognition pipeline is implemented as a hybrid C/CUDA and Python application that 
runs on Google Colab environments. The core recognition algorithm processes grayscale PGM 
images by first extracting LBP features, which create a spatial histogram representation of facial 
textures. 
The implementation follows a traditional supervised learning approach using the K-Nearest 
Neighbors algorithm where images from known individuals are processed to create a 
reference database. When presented with an unknown face image, the system computes its 
LBP histogram and identifies the person by finding the K most similar histograms using the 
chi-square distance metric, then determining the identity through majority voting among these 
neighbors.The architecture includes a GPU-accelerated implementation with CPU fallback 
mechanisms to ensure reliability. 
During training, the application processes a dataset organized by person IDs, extracting and 
storing LBP histogram features. Later, accuracy is computed with the test images available. 
During prediction, the image that is input is processed similarly and compared against all 
reference histograms to determine the best match. 
An interactive Google Colab interface provides user-friendly access to the recognition 
capabilities, handling image upload, format conversion, and result display.

**TECHNICAL HIGHLIGHTS**

● Feature Extraction: Implements 8-bit Local Binary Pattern operators for texture 
analysis. 

● CUDA Parallel Processing: Utilizes grid-based thread parallelism to accelerate image 
processing with a 16×16 thread block configuration. 

● Hybrid Computation Model: Incorporates seamless fallback to CPU processing when 
GPU resources are unavailable or CUDA operations fail 

● Optimized Memory Management: Implements efficient memory handling for both host 
and device operations, with appropriate flattening of 2D structures for GPU compatibility

● KNN Classification Model: Implements a KNN classifier to classify the input images. 

● Chi-Square Distance Metric: Employs histogram comparison optimized for texture 
feature matching in facial recognition contexts

● Atomic Operations: Uses CUDA atomic operations to handle concurrent histogram 
updates during parallel processing 

● Histogram Normalization: Applies proportional scaling to ensure consistent comparison 
between images of different sizes 

● Interactive Interface: Provides a web-based user interface.
