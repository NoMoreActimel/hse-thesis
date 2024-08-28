# Developing effective StyleGAN encoders for domain adaptation
### This is my undergraduate thesis at HSE, the final text is in the [sedov-hse-thesis.pdf](sedov-hse-thesis.pdf)!

### Initially we were working on the [StyleDomain repository](https://github.com/FusionBrainLab/StyleDomain)

Starting from the [StyleDomain paper](https://arxiv.org/abs/2212.10229) results, 
we have focused on the improvement of interaction between the StyleSpace domain adaptation method and StyleGAN encoders,
such as [Encoder-4-Editing](https://arxiv.org/pdf/2102.02766) and [Feature-Style-Encoder](https://arxiv.org/abs/2202.02183).

As the result, we have trained the FeatureShift domain encoder that unites knowledge of pretrained StyleSpace weights over 70 domains 
and improves reconstruction/adaptability trade-off being used along with more accurate Feature-Style-Encoder.
