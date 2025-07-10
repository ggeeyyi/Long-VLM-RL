# LongVideo-Reason

## Data Preparation
You can find the videos for testing [here](https://huggingface.co/datasets/LongVideo-Reason/longvideo_eval_videos/tree/main). Please download them, and `tar -zxvf` them into a directory named `longvila_videos`.
```
├── $VIDEO_DIR
│   ├── longvila_videos
│   │   │── mp4/webm/mkv videos
```

## Testing
`$VIDEO_DIR` is the parent directory of `longvila_videos`. For different models, you need to customize the `model_generate` function accordingly. The model generations and output metrics will be saved in `runs_${$MODEL_PATH}`.
```bash
python3 eval.py \
        --model-path $MODEL_PATH \
        --data-path LongVideo-Reason/longvideo-reason@test \
        --video-dir $VIDEO_DIR \
        --output-dir runs_${$MODEL_PATH}
```

## Citation
Please consider to cite our paper if this benchmark are helpful in your research.

```bibtex
@article{chen2025longvila-r1,
      title={Scaling up Long Video Reasoning},
      author={Yukang Chen and Wei Huang and Baifeng Shi and Qinghao Hu and Hanrong Ye and Ligeng Zhu and Zhijian Liu and Pavlo Molchanov and Jan Kautz and Xiaojuan Qi and Sifei Liu and Hongxu Yin and Yao Lu and Song Han},
      year={2025},
      eprint={xxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```