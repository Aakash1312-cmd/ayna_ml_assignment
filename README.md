Insights Report: Conditional Polygon Colorization

Hyperparameters
Final Settings:
    a.Epochs: 100.
    b.Batch Size: 16.
    c.Learning Rate: 1e-4.
    d.Image Size: 128x128 pixels.
    e.Optimizer: Adam.

The ultimate hyperparameter values were selected to support a robust training process on free-tier Kaggle GPUs. A starting learning rate of 1e-3 resulted in unstable loss, so it was set to 1e-4 for smoother convergence. A batch size of 16 was chosen as a good size to fit within the memory limits of the hardware available while supporting efficient training. These values were effective for the model to learn the task for 100 epochs.

Architecture
UNet design/conditioning choices:
    The final architecture employed was the UNet2DConditionModel from the Hugging Face Diffusers library.
    This model was specifically chosen for its integrated cross-attention blocks (CrossAttnDownBlock2D, CrossAttnUpBlock2D), which are designed for robust conditional generation.
The color conditioning was done by one-hot encoding the input color name (e.g., "blue"). The encoded vector was then passed to the encoder_hidden_states argument of the model, so that the cross-attention layers were able to direct the image generation effectively from the selected color.

Any ablations:
A from-scratch ConditionalUNet was also used initially, where the embedding of color was inserted directly into the model's bottleneck layer. The project eventually shifted to the more complex UNet2DConditionModel because its cross-attention mechanism provided a stronger and more accurate way to inject conditional information at various stages of the U-Net.

Training dynamics
Loss/metric curves, Qualitative Output Trends
Training was monitored through Weights & Biases in the project "conditional-UNet-polygons."
The most important metrics to monitor were L1 Loss and Peak Signal-to-Noise Ratio (PSNR). During training, the loss decreased steadily while PSNR increased consistently, signifying effective learning.
Qualitatively, the output of the model evolved from producing washed-out or erroneous colors during early epochs to vibrant, correct colors with clean polygon edges during later epochs.

Common failure modes, and fixes tried**:
    1.  Failure Mode: Inference error in color generation (e.g., asking for green resulted in yellow).
Fix: This was a critical bug that was fixed by making sure the inference logic for color conditioning (one-hot encoding) exactly mimicked the logic applied within the training loop.
2.  Failure Mode: FileNotFoundError when executing the inference notebook in a new session.
Cause: The issue was traced to erroneous file paths. The code was refreshed with the proper, absolute paths in the Kaggle environment, namely /kaggle/input/ayna-dataset-full/dataset for the data and /kaggle/input/best-model-diffusers-2-0 for the stored model.
3.  Failure Mode: A loading error for the model citing diffusion_pytorch_model.bin not found.
This was due to the fact that the weight file of the model had a non-standard name (diffusion_pytorch_model-2.safetensors). The solution was to make the loading more resilient by first initializing the model structure based on the config.json and then manually loading the weights from the particular .safetensors file.

Learnings:

Most important to learn was keeping spot-on consistency between the training and inference data pipes. Even a tiny inconsistency in preprocess or model input format can cause serious output failures.
When operating within cloud platforms such as Kaggle, it's necessary to employ strong, explicit file paths. Hardcoded relative paths are fragile and will shatter instantly between sessions.
Programmatic model loading (constructing from a config file, followed by loading a state dictionary from a particular weights file) is a more stable method compared to relying on a library's default file-finding magic, particularly when working with non-standard filenames.
For submission and sharing, it is important to make use of the site's capabilities properly, e.g., publishing public Kaggle Datasets of model weights and employing "Save & Run All" to create a shareable notebook version with all outputs accessible.
