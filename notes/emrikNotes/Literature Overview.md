
### Simon
Thesis investigates the problem of large scale vehicle data readout and presents a solution to it. The solution leverages a lossless streaming based compression at low cost. Suggests the requirements for a single vehicle in terms of compute and memory. One problem is that the data comes in various formats. It is a challenges to balance accuracy and compression. Electrical Control Units are used to collect data for a single component in the car and their architecture resembles that of a normal computer with CPU, RAM, flash memory and I/O bridges. The use case for the data is probably for AI training. As  data grows, the amount of data needed to be stored exceeds that of which a relational data base would be capable of. 

The problem can be divided into the receiver problem and the producer problem where the producer problem is that trucks have to send a certain amount of data in a small enough format that the bandwidth can communicate it. The receiver problem on the other hand is that data is being streamed to one endpoint from thousands of different vehicles placing considerable strain on the endpoint. 

Runs GORILLA AND CHIMP algorithm for benchmarking. Gives times and compression ratios for the data. 

### Normalizing flow: an introduction and review of the current methods
https://arxiv.org/abs/1908.09257

variational autoencoder

Shanonns source coding theorem

### Denoising Diffusion Probabilistic Models
https://arxiv.org/abs/2006.11239

generative adversarial networks

Vimeo-90k datase
### Lossy Image Compression with Conditional Diffusion Models
https://arxiv.org/abs/2209.06950

### Efficient Lossy Compression of Video Sequences of Automotive high-Dynamic Range Image Sensors for Advanced Driver-Assistance Systems and Autonomous Vehicles

Suggests that lossless compression is used for automotive industry for security reasons, but makes the compression rate worse. 

https://www.mdpi.com/2079-9292/13/18/3651

- lossless compression
- existing reaserch about data compression on vehichels

Whether a neural codec is slower or faster depends on the architecture (autoregressive vs parallel), weather you use a gpu and other engineering choices. (pruning, quantization, memory I/O optimization) [link](https://arxiv.org/html/2502.20762v1?utm_source=chatgpt.com)


### Rethinking Learned Image Compression: Context is All You Need

Problems with existing neural codecs is their slower encoding speed which usually is caused by having computationally heavy networks and sequential processing of symbols [link](https://arxiv.org/html/2407.11590v1?utm_source=chatgpt.com). The neural compression method is getting closer to traditional methods but still struggle with 

The problem with traditional methods is that they now struggle with meeting modern requirements for lower bit rates and higher objective and subjective image quality. 

Another prominent issue is the operational cost of memory movement, many small kernels / function calls and cache behavior. 

Challenges with implementing learned codecs on embedded systems such as trucks:
- latency
- predictability
- memory footprint
- robustness
- compression ratios (not really)

### DeepN-JPEG A deep neural network favorable JPEG-based image compression framework

"An example of how a JPEG-based image compression framework can be used to compress image data for image recognition with a 3.5x higher compression ratio."[Link](https://arxiv.org/abs/1803.05788) Shows that image compression can lower accuracy while still being very clear to humans. 

### Learned Image Compression for Machine Perception (2021)
Compression using codecs have to balance image distortion and compression ratio. "the traditional goal of lossy image compression is to find the best trade-off between minimizing the expected length of the bit-stream representation of the input and minimizing the expected distortion between the image and its reconstruction" (rate distortion problem). $$R(Q(f(x)))+\lambda*D(g(Q(f(x))), x)$$ The R represent the bit-stream length and D represent the distance between the original image and the reconstructed image. $f(x)$ represent the discrete representation of the input x. Q is the quantization operator. We cannot minimize both tasks simultaneously and therefore use $\lambda$ to balance the tradeoff. Visual tasks can be performed on the representation $f(x)$ without reconstructing it. Hypothesis is that there exists many representations which achieve the same level of rate-distortion with different suitability for different machine learning tasks. They propose a different loss function which considers some kind of machine task: $$R(Q(f(x)))+\lambda_d*D(g(Q(f(x))), x)+\lambda_u*U(Q(f(x)))$$
Here U is some pragmatically defined utility based on the loss function of the machine task. They test a number of different computer vision tasks and to see if the compression algorithm can be biased to preserve semantic information and how well the information generalizes to other tasks. The paper concludes that the representations they produce using this technique is effective both for low shot learning regime and when training models for new tasks directly on the compressed representations. 

### Model Compression for Deep Neural Networks: A Survey

Ideas for how to compress neural networks as to make them fit on embedded hardware. Suggests using hybrid precision in order to reduce model size while preserving performance and neural architecture search. [Link](https://www.mdpi.com/2073-431X/12/3/60)

### TARNet : Task-Aware Reconstruction for Time-Series Transformer

[Link](https://dl.acm.org/doi/pdf/10.1145/3534678.3539329)
Perhaps not super relevant to the project but the TARNet works by having a biased autoencoder which reconstructs data based on how important it is for the downstream task. Traditional autoencoders treat all points in the time series equally but this often leads to the decoder reconstructing unnecessary noise. In this model, the reconstruction loss is a combination of MSE and the importance for the timepoint: $$L_TAR = Σ α_t * |X_t - X̂_t|^2$$
Here the $\alpha_t$ is learned through gradient decent of the downstream task. The total loss will then be: $$L_{total} = L_{task} + λ * L_{TAR}$$

This model balances the ability to reconstruct the timeseries with the ability to perform tasks such as prediction. We could apply this by incorporating the bit stream length in the loss function as to prioritize compression even more. 

### Time Series Compression Survey (2023)

[Link]([https://doi.org/10.1155/2023/5025255](https://doi.org/10.1155/2023/5025255))

Using the compressed data as input for classification tasks is one of the main research gaps identified in the review. The paper points out the possibility that the computational capabilities of edge devices which collect data likely will become more powerfull in the future, making ML based compression more viable. 

### Extract, Compress and Encode: LitNet an Efficient Autoencoder for Noisy Time-Series Data

[Link](https://ieeexplore.ieee.org/document/10002775/authors#authors)
According to ChatGPT:
focuses on noisy time-series data and producing a compact encoding while preserving important features so that downstream tasks still work well. The architecture is multilevel encoder/decoder with small filters in stack, aiming for a lightweight design. The evaluation metric is feature reconstruction and latent space separator. 

It is slightly different from what we would do as it does not incorporate the loss in a way where it allows it to affect the compression. The compression ratio is also not presented, only the accuracy and parameter count. The data used in this paper is also quite limited and does not give a great idea of how the model would function in production. 

### Autonomous Internet of Things (IoT) Data Reduction Based on Adaptive Threshold
[Link](https://www.mdpi.com/1424-8220/23/23/9427)

Designs a system which varies the amount of data transmitted based on a adaptive threshold. This helps reduce data transmission while maintaining or increasing accuracy of data reconstruction. Uses CUSUM to detect changes in the signal and send more or less data. uses a Kalman filter to detect trends in the signal. 

Very lightweight but could be extended by replacing the Kalman filter with a small learned predictor with a threshold which is optimized for downstream tasks. Also using a codec to compress the data before sending it. 

### Highly Efficient Direct Analytics on Semantic-aware Time Series Data Compression
[Link](https://arxiv.org/pdf/2503.13246)

The goal of this paper is very similar to ours as they present a model which is able to compress data with focus on semantic preservation. It is however not a neural compression approach as it does not use a model with loss to optimize the compression. They also only test for outlier detection. 