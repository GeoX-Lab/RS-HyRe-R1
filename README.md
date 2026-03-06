# RS-HyRe-R1: A Hybrid Reward Mechanism to Overcome Perceptual Inertia for Remote Sensing Images Understanding

In recent years, Remote Sensing Vision-Language Models (RS-VLMs) have demonstrated immense potential in remote sensing (RS) tasks, with the introduction of Reinforcement Learning (RL) post-training further expanding their performance boundaries. However, the complexity of remote sensing imagery (RSI) necessitates exhaustive visual scanning for accurate interpretation. Although RL-trained RS-VLMs exhibit significant performance gains over base models, observations reveal that their inference processes often rely on minimal salient visual cues (e.g., large color blocks or distinct contours) to rapidly fit answers. We term this phenomenon RL-induced  “Perceptual Inertia",  where models, driven by the greedy nature of reward maximization, tend to converge onto a single ”most efficient path". This greedy optimization results in dual deficits: cognitive path homogenization, where reliance on specific features hinders complete evidence construction, and executive inflexibility, causing localization errors during task switching.
To overcome this optimization bias, compel the model to resume exhaustive mining of visual evidence, and enhance cross-task generalization, we propose RS-HyRe-R1, a hybrid reward mechanism for RS images understanding. Specifically, this mechanism utilizes spatial reasoning activation reward to impose structural constraints for explicit visual reasoning. 
To ensure the model flexibly focuses on different visual information across tasks, the RS-task perception correctness reward establishes adaptive quality anchors. By dynamically calibrating answer deviations across different task paradigms, it ensures precise locking of geometric boundaries and semantic ground truths during during transitions between RS tasks. Furthermore, visual-semantic path evolution reward is integrated to penalize repetitive reasoning patterns, thereby incentivizing the capture of neglected complementary cues to construct comprehensive evidence chains.
Experimental results confirm that RS-HyRe-R1 effectively reverses the ``perceptual inertia",  prompting the model to generate deep reasoning sequences rich in diverse visual detail. With only 3B parameters, it achieves state-of-the-art performance across referring expression comprehension (REC), open vocabulary object detection (OVD), and Visual Question Answering (VQA) tasks, outperforming existing models with up to 7B parameters. Notably, in cross-dataset zero-shot testing, it improves VQA accuracy and REC localization precision by 3.16\% and 3.72\%, respectively, over the second-best model, demonstrating superior generalization capabilities. The source code and dataset are available at https://github.com/geox-lab/RS-HyRe-R1.

## Perceptual Inertia

![abstract](D:\Doctoral\Paper\First_paper\write\RS-HyRe-R1\abstract.png)

Fig1：Illustration of the “Perceptual Inertia'” phenomenon and our proposed RS-HyRe-R1 solution.
(a) Existing RL-driven models (e.g., TinyRS-R1, Geo-R1) often struggle with visual reasoning, exhibiting hallucinations or relying on superficial feature matching.  (b) We identify this bottleneck as “Perceptual Inertia”—the tendency of models to fixate on salient shortcuts (e.g., buildings) while neglecting broader context. (c) To overcome this, our RS-HyRe-R1 employs a hybrid reward evolutionary strategy that incentivizes ``Observing More Diverse''. This strategy constructs diverse and logically rigorous reasoning chains by observing more complementary visual evidence.

## Method

![method](D:\Doctoral\Paper\First_paper\write\RS-HyRe-R1\method.png)

Fig2：Overview of the proposed RS-HyRe-R1 framework. The pipeline begins with the construction of a RS-Task Dataset (Left), utilizing only 1,600 samples across REC, OVD, and VQA tasks. We employ GRPO for RL training (Middle), where the Policy Model generates multiple rollout answers (o_1 to o_N). The optimization is guided by the Hybrid Reward Mechanism, which evaluates outputs through three dimensions: Spatial Reasoning Activation, RS-task Perception Correctness, and Visual-Semantic Path Evolution. The rightmost section demonstrates the model's capability to generate comprehensive reasoning chains and accurate predictions across all three tasks.

## Results

![REC_5](D:\Doctoral\Paper\First_paper\write\RS-HyRe-R1\REC_5.png)

![OVD_5](D:\Doctoral\Paper\First_paper\write\RS-HyRe-R1\OVD_5.png)
