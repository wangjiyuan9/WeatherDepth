<div align="center">
<h1>WeatherDepth: Curriculum Contrastive Learning for Self-Supervised Depth Estimation under Adverse Weather Conditions</h1>

<div>
    <a href='https://scholar.google.com/citations?user=subRjlcAAAAJ&hl=zh-CN' target='_blank'>Jiyuan Wang</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=t8xkhscAAAAJ' target='_blank'>Chunyu lin</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=vo__egkAAAAJ' target='_blank'>Lang Nie</a><sup>1</sup>&emsp;
    <a href='XXX' target='_blank'>Shujun huaang</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>Beijingjiaotong University
</div>



<div>
    <h4 align="center">
        • <a href="https://arxiv.org/abs/2310.05556" target='_blank'>ICRA 2024</a> •
    </h4>
</div>


## Abstract

<div style="text-align:center">
<img src="assets/Figure3.pdf"  width="80%" height="80%">
</div>

</div>
<strong>Depth estimation models have shown promising performance on clear scenes but fail to generalize to adverse weather conditions due to illumination variations, weather particles, etc. In this paper, we propose WeatherDepth, a self-supervised robust depth estimation model with curriculum contrastive learning, to tackle performance degradation in complex weather conditions. Concretely, we first present a progressive curriculum learning scheme with three simple-to-complex curricula to gradually adapt the model from clear to relative adverse, and then to adverse weather scenes. It encourages the model to gradually grasp beneficial depth cues against the weather effect, yielding smoother and better domain adaption. Meanwhile, to prevent the model from forgetting previous curricula, we integrate contrastive learning into different curricula. By drawing reference knowledge from the previous course, our strategy establishes a depth consistency constraint between different courses toward robust depth estimation in diverse weather. Besides, to reduce manual intervention and better adapt to different models, we designed an adaptive curriculum scheduler to automatically search for the best timing for course switching. In the experiment, the proposed solution is proven to be easily incorporated into various architectures and demonstrates state-of-the-art (SoTA) performance on both synthetic and real weather datasets.</strong>

---

</div>
The source code is comming!
