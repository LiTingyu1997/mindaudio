# Speaker verification - Use ECAPA-TDNN to extract speaker identification

## Requirements
| mindspore     |   ascend driver        | firmware     |  cann toolkit/kernel    |
|:-------------:|:----------------------:|:------------:|:-----------------------:|
|     2.0.0     |   23.0.RC2             | 6.4.12.1.241 |  6.3.RC1                |

## Introduction

ECAPA-TDNN was proposed by Desplanques et al., Goth University, Belgium, in 2020. By introducing the SE (squeeze excitation) module and channel attention mechanism, this model has won the first place in the VoxSRC2020.

### Model Architecture

In view of some advantages and disadvantages of the current X-Vector-based voice print recognition system, ECAPA-TDNN has improved from the following three aspects:

**1、Channel- and context-dependent statistics pooling**

**2、1-Dimensional Squeeze-Excitation Res2Block**

**3、Multi-layer feature aggregation and summation**

![tdnn.png](https://github.com/mindspore-lab/mindaudio/blob/main/tests/result/tdnn.png?raw=true)

### Data Processing

- Audio:

  1.Feature extraction: fbank

  2.Data augmentation：add_babble, add_noise, add_reverb, drop_chunk, drop_freq, speed_perturb。

     Current accuracy can be obtained using 5X data enhancement (which requires 2.6 terabytes of disk space). If you want to achieve EER(0.8%), you need 50x data enhancement, just change the hyperparameter 'number of epochs' in the 'ecapatdnn.yaml' file to 10 (50x data enhancement requires 26T disk space).

## Usage Steps

### 1. Preparing Dataset（VoxCeleb1 + VoxCeleb2 ）

Voxceleb2 audio files are released in m4a format. All the files must be converted in wav files before training. Please, follow these steps to prepare the dataset correctly:

1. Download both Voxceleb1 and Voxceleb2.
You can find download instructions here: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
Note that for the speaker verification experiments with Voxceleb2 the official split of voxceleb1 is used to compute EER.

2. Convert .m4a to wav
Voxceleb2 stores files with the m4a audio format. To use them within MindAudio you have to convert all the m4a files into wav files.
You can do the conversion using ffmpeg(https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830). This operation might take several hours and should be only once.

3. Put all the wav files in a folder called wav. You should have something like `voxceleb12/wav/id*/*.wav` (e.g, `voxceleb12/wav/id00012/21Uxsk56VDQ/00001.wav`)

4. copy the `voxceleb1/vox1_test_wav.zip` file into the voxceleb12 folder.

5. Unpack voxceleb1 test files(verification split)， Go to the voxceleb2 folder and run `unzip vox1_test_wav.zip`.

6. copy the `voxceleb1/vox1_dev_wav.zip` file into the voxceleb12 folder.

7. unpack voxceleb1 dev files, go to the voxceleb12 folder and run `unzip vox1_dev_wav.zip`.

8. Unpack voxceleb1 dev files and test files in dir `voxceleb1/`. You should have something like `voxceleb1/wav/id*/*.wav`.

9. Enhance the need to use the data ` rirs noises.zip ` files, can be downloaded in this link: :http://www.openslr.org/resources/28/rirs_noises.zip, then put it in ` voxceleb12 / ` directory.

### 2. Training

#### Single-Card Training

When the Voxceleb1 and Voxceleb2 datasets are ready, run the following script to pre-process the audio data and train the speaker's signature on a single card:

```shell
# Standalone training
python train_speaker_embeddings.py
```


Voxceleb1 and Voxceleb2 data sets are large, and it takes a long time to preprocess audio data. Therefore, 30 processes are used for audio preprocessing at the same time.

modify the `data_process_num` in `ecapatdnn.yaml` for debugging.

After the preprocessed data is generated, run the following code for single-card training:

```shell
# Standalone training with prepared data
python train_speaker_embeddings.py --need_generate_data=False
```

#### Multi-Card Training

After the preprocessed data is generated, run the following code for distributed multi-card training:

`bash ./run_distribute_train_ascend.sh hccl.json`

hccl.json  is generated using hccl tool, refer to this article tools (https://gitee.com/mindspore/models/tree/master/utils/hccl).

### 3.Eval

After model training is complete, run the following script for verification:

```shell
# eval
python speaker_verification_cosine.py
```

If preprocessed data is generated, set `--need_generate_data=False`:

```shell
# eval with prepared data
python speaker_verification_cosine.py --need_generate_data=False
```



## **Performance**

Experiments are tested on ascend 910 with mindspore 2.0.0 graph mode:

| model name | cards | batch size | s/step | recipe | weight | eer | eer with s-norm |
|:----------:|:-----:|:----------:|:------:|:------:|:------:|:---:|:---------------:|
| ecapa-tdnn |   8   |   32       |  0.38  | [yaml](https://github.com/mindspore-lab/mindaudio/blob/main/examples/ECAPA-TDNN/ecapatdnn.yaml) | [weights](https://download.mindspore.cn/toolkits/mindaudio/ecapatdnn/ecapatdnn_vox12.ckpt)| 1.50% | 1.69%  |
