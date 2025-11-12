# Generative Model Validation and Distribution Fidelity

A critical requirement for successful SSH inpainting is ensuring that the Flow Matching model has learned the true data distribution $p_{\text{data}}(\mathbf{x})$ of SSH fields. Poor distribution matching can lead to generated samples that exhibit unrealistic oceanographic features, incorrect energy spectra, or biased statistical properties that compromise reconstruction quality.

We unconditionally sampled 64 samples per lat/lon map pairs to compare the generated data distribution with the real data distribution. In this experiment, we used different ODE solvers.

## Pixel by pixel

We perform a quantitative assessment of marginal distribution alignment by comparing the pixel-wise statistics of generated and real SSH patches. Let $\mathcal{D}_{\text{real}}$ denote a collection of real SSH patches from the test set, and $\mathcal{D}_{\text{gen}}$ represent a set of patches sampled from the trained Flow Matching model.

For statistical comparison, we compute empirical probability density functions and evaluate their similarity using multiple metrics:

**Kolmogorov-Smirnov Test** to assess whether the pixel value distributions are drawn from the same underlying distribution:

$$
D_{KS} = \sup_x | F_{\text{real}}(x) - F_{\text{gen}}(x) |
$$

with $F_{\text{real}}$ and $F_{\text{gen}}$ the empirical cumulative distribution functions.

**1-Wasserstein Distance** provides a metric for distribution discrepancy:

$$
W_1(P_{\text{real}}, P_{\text{gen}}) = \inf_{\gamma \in \Gamma(P_{\text{real}}, P_{\text{gen}})} \mathbb{E}_{(x,y) \sim \gamma}\|x-y\|
$$

**Table 1: Statistical comparison of real vs synthetic distributions (pixel-level)**
*Reported are KS statistic D and p-value, and Wasserstein-1 distance.*

| ODE solver (steps) | euler(50) | heun2(50) | heun3(50) | rk4(10) | rk4(50) | dopri5 | dopri8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| KS statistic D | 0.054 | **0.025** | 0.026 | 0.027 | 0.043 | 0.047 | 0.047 |
| KS p-value | 6.6×10⁻⁴ | **0.378** | **0.328** | 0.271 | 0.013 | 0.0051 | 0.0050 |
| Wasserstein-1 | 0.097 | 0.056 | 0.052 | **0.039** | 0.089 | 0.073 | 0.052 |

Table 1 analyses the FM model's generated SSH probability laws versus the true empirical distribution of measurements, albeit marginally. The Heun2 and Heun3 solvers' Kolmogorov-Smirnov test results show the best outcomes, meaning that these solvers yield the least locally deviated marginal cumulative density functions. Heun2/3 and RK4 show a KS p-value >5%, meaning that we cannot statistically reject the hypothesis of equality between the empirical and generated cumulative distributions. The Runge-Kutta 4 solver, with 10 steps of integration, provides the best Wasserstein-1 distance, meaning that the generated PDF is the closest in the sense of $W_1$ (Earth mover's distance). However, RK4's cumulative density function has worse local deviations with respect to the true CDF than Heun2/3, as shown by its non-optimal KS results.

## Adversarial Discrimination Analysis

We employ a binary classification framework to evaluate the indistinguishability of generated samples from real data. A convolutional neural network discriminator $D_\phi: \mathbb{R}^{H \times W} \to [0,1]$ is trained to classify patches as real ($y=1$) or generated ($y=0$) using the standard binary cross-entropy loss:

$$
\mathcal{L}_{\text{disc}}(\phi) = -\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}\log D_\phi(\mathbf{x}) - \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\text{gen}}}\log(1-D_\phi(\tilde{\mathbf{x}}))
$$

We employ a ResNet18 architecture (He et al., 2016) as the discriminator backbone, modified for binary classification by replacing the final fully connected layer with a single-output layer followed by sigmoid activation. The ResNet18 provides robust feature extraction through its residual connections while maintaining computational efficiency for SSH patch classification. We train on balanced datasets with $N_{\text{real}} = N_{\text{gen}}$ patches using Adam optimization with a learning rate of $1 \times 10^{-3}$.

We evaluated discrimination quality using a standard classification metric, AUC (Area Under Curve), that should ideally be close to 0.5. We also analyzed the discriminator's prediction confidence distributions for both real and synthetic samples.

Real confidence = $\mathbb{E}[D_\phi(\mathbf{x})]$ for $\mathbf{x} \sim p_{\text{data}}$  
Synthetic confidence = $\mathbb{E}[D_\phi(\tilde{\mathbf{x}})]$ for $\tilde{\mathbf{x}} \sim p_{\text{gen}}$

High-quality generative models should exhibit overlapping confidence distributions with similar means and standard deviations, low discriminator accuracy indicating difficulty in distinguishing real from generated samples, and balanced precision-recall performance across both classes.

**Table 2: ResNet-18 performance on distinguishing generated data from real**
*Confidence and Predicted Ratio measured on generated data only.*

| ODE solver (steps) | euler(50) | heun2(50) | heun3(50) | rk4(10) | rk4(50) | **dopri5** | dopri8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AUC | 0.998 | 0.968 | 0.539 | 0.747 | 0.518 | **0.516** | 0.538 |
| Mean Confidence (Real) | 0.018 | 0.106 | **0.500** | 0.163 | 0.499 | 0.501 | 0.499 |
| Mean Confidence (Gen) | 0.982 | 0.894 | **0.500** | 0.837 | 0.501 | 0.499 | 0.501 |
| Predicted ratio (Real / Gen) | 1.6% | 10.4% | 66.6% | 16.3% | 49.3% | 59.5% | **51.2%** |

Results are reported in Table 2, the best compromise is achieved by the dopri5 solver as it yields the AUC closest to 1/2 and a discriminator confidence close to 1/2. Moreover, in Figure 1, we highlight that we failed to train a reliable discriminator on real and DOPRI5 generated samples, which is not the case when using the Euler solver.

**Figure 1: Classifier distributions for different ODE solvers**

![Classifier distribution for DOPRI5](images/dist_ver_classifier_dopri5.png)
*Classifier distribution for DOPRI5*

![Classifier distribution for Euler 50 steps](images/dist_ver_classifier_euler50.png)
*Classifier distribution for Euler 50 steps*

*Normal distributions fitted to real (blue) and generated (orange) confidence histograms. Left: DOPRI5 with overlapping distributions, close means (0.501 vs 0.499), and low accuracy (51.2%). Right: Euler 50 steps with separated distributions, distant means (0.018 vs 0.982), and high accuracy (98.4%).*

## Latent Space Consistency Validation

We assess distribution fidelity using a convolutional autoencoder trained exclusively on real SSH data to learn a compressed latent representation that captures the essential structural properties of oceanographic fields. The autoencoder architecture consists of an encoder $E_\psi: \mathbb{R}^{H \times W} \to \mathbb{R}^d$ and decoder $D_\psi: \mathbb{R}^d \to \mathbb{R}^{H \times W}$ trained to minimize the reconstruction loss:

$$
\mathcal{L}_{\text{AE}}(\psi) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\|\mathbf{x} - D_\psi(E_\psi(\mathbf{x}))\|_2^2]
$$

The encoder employs a hierarchical convolutional architecture with two stages of convolution-pooling operations (32→64→128 channels), followed by spatial downsampling via 2×2 max pooling. This network encodes 24×24 patches into 6×6×128 feature representations before projection to a latent dimension. The decoder uses transposed convolutions to reconstruct the original patch dimensions, with batch normalization and LeakyReLU activations throughout both encoder and decoder networks.

For both real and generated patches, we compute mean squared reconstruction errors:

$$
\mathcal{E}_{\text{real}} = \{\|\mathbf{x}_i - D_\psi(E_\psi(\mathbf{x}_i))\|_2^2 : \mathbf{x}_i \in \mathcal{D}_{\text{real}}\}
$$

$$
\mathcal{E}_{\text{gen}} = \{\|\tilde{\mathbf{x}}_j - D_\psi(E_\psi(\tilde{\mathbf{x}}_j))\|_2^2 : \tilde{\mathbf{x}}_j \in \mathcal{D}_{\text{gen}}\}
$$

High-quality generative models should exhibit: 1) similar mean reconstruction errors indicating comparable latent space positioning, 2) comparable error variance suggesting consistent reconstruction difficulty, 3) non-significant statistical tests (p > 0.05) indicating distributional equivalence, and 4) small effect sizes (|Cohen's d| < 0.2, |Cliff's δ| < 0.147) demonstrating minimal practical differences.

In Table 3, we report the real versus synthetic distributional differences for the autoencoder reconstruction error. Runge-Kutta 4 generated samples are the best candidates, as they yield the lowest Kolmogorov-Smirnov distance with the highest p-value, as well as the lowest Wasserstein-1 distance. Overall, the synthetic reconstruction error distribution approaches the real one as the number of inferences increases for the ODE integration. However, the p-value of the rk4 samples is less than 5%, meaning that the two distributions can actually be separated using KS test. In addition, we report the histogram plot of the probability density function for real versus generated autoencoder reconstruction errors. Note that Figure 2 shows a similar PDF; whereas, when generating using the Euler solver instead of Heun2, the synthetic reconstruction error distribution is further from the observation reconstruction error distribution.

**Table 3: Distribution comparison of reconstruction errors (autoencoder verification)**
*Reported are mean values for synthetic - real MSEs, KS statistic D and p-value, and Wasserstein-1 distance. Best values are in bold.*

| ODE solver (steps) | euler(50) | heun2(50) | heun3(50) | rk4(10) | rk4(50) | dopri5 | dopri8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Error deviation | -0.0025 | **0.0003** | 0.0009 | -0.0006 | 0.0005 | 0.0011 | 0.0008 |
| KS D | 0.0290 | 0.0869 | 0.0338 | 0.1257 | **0.0230** | 0.0261 | 0.0283 |
| KS p-value | 0 | 0 | 0 | 0 | **0.0101** | 0.00220 | 0.00066 |
| Wasserstein-1 | 0.00252 | 0.00092 | 0.00112 | 0.00171 | **0.000807** | 0.00115 | 0.00106 |

**Figure 2: Auto-encoder reconstruction errors distributions for different ODE solvers**

![Auto-encoder reconstruction errors distribution for Euler 50 steps](images/dist_ver_ae_heun2_50.png)
*Auto-encoder reconstruction errors distribution for Euler 50 steps*

![Auto-encoder reconstruction errors distribution for rk4 50 steps](images/dist_ver_ae_rk4_50.png)
*Auto-encoder reconstruction errors distribution for rk4 50 steps*

*Real distributions fitted to real (blue) and generated (red) pixel value histograms. Top: Euler 50 steps with significant distribution mismatch (KS D=0.029). Bottom: rk4 50 steps with close alignment (KS D=0.023) indicating no significant difference.*

## Section conclusion

This multifaceted approach provides robust validation by detecting subtle marginal, separating, or reconstruction distributional inconsistencies that may not be apparent through visual inspection or pixel-level comparisons, ensuring that generated SSH fields exhibit authentic statistical properties consistent with real oceanographic data.

# Reference

**[He et al., 2016]** Deep Residual Learning for Image Recognition, *Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun*, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pages 770-778, https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
