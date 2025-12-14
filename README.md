# SAR Image Processing — VST Demo

Interactive demo of a Variance Stabilization Transform (VST) pipeline for speckle noise reduction in synthetic SAR-like images.

**Notebook:** `sample_interactive.ipynb` — an interactive widget-based notebook that lets you tweak VST parameters, noise level and filtering settings.

**Requirements**
- Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

**Run the Notebook**
- Start Jupyter and open the notebook (or open it directly in VS Code):

```bash
jupyter notebook sample_interactive.ipynb
# or
jupyter lab sample_interactive.ipynb
```

**Screenshot**

![image](assets\VST_example.png)

**Transforms**

The notebook implements a parametric Variance Stabilization Transform (VST). The forward transform applied to image intensities $I$ is:

$$
T(I) = a \log_b I = \frac{a \ln I}{\ln b}
$$

The inverse transform (mapping back to intensity values) is:

$$
T^{-1}(y) = b^{y / a}
$$

When converting multiplicative speckle noise with standard deviation $\sigma_{\text{mult}}$ to an additive-equivalent standard deviation in the transformed domain, the notebook uses:

$$
\sigma_{\text{add}} = \frac{a\,\sigma_{\text{mult}}}{\ln b}
$$

Variables: $a$ — scale parameter, $b$ — logarithm base, $I$ — intensity, $y$ — transformed value.

