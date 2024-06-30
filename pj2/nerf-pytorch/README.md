# NeRF-pytorch

## 安装环境

```
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt
```

<details>
  <summary> Dependencies  </summary>
  
  ## Dependencies
  - PyTorch 1.4
  - matplotlib
  - numpy
  - imageio
  - imageio-ffmpeg
  - configargparse
  
如果需要加载LLFF数据，还需要安装ImageMagick。

为了运行自己的数据，还需要设置LLFF代码（以及COLMAP）来计算姿态，具体请参考报告
  
</details>

## 示例代码运行

### Quick Start

下载两个示例数据集：lego 和 fern

```
bash download_example_data.sh
```

训练一个低分辨率的lego NeRF：
```
python run_nerf.py --config configs/lego.txt
```
训练100k次迭代后（约4小时，使用一块2080 Ti），可以在logs/lego_test/lego_test_spiral_100000_rgb.mp4找到以下视频。

![](https://user-images.githubusercontent.com/7057863/78473103-9353b300-7770-11ea-98ed-6ba2d877b62c.gif)

---

训练一个低分辨率的fern NeRF：

```
python run_nerf.py --config configs/fern.txt
```
训练200k次迭代后（约8小时，使用一块2080 Ti），你可以在logs/fern_test/fern_test_spiral_200000_rgb.mp4 和 logs/fern_test/fern_test_spiral_200000_disp.mp4找到以下视频。


![](https://user-images.githubusercontent.com/7057863/78473081-58ea1600-7770-11ea-92ce-2bbf6a3f9add.gif)

---

### 目录结构
放置自己的数据集时，需要参考下面目录结构：
```
├── configs                                                                                                       
│   ├── ...                                                                                     
│                                                                                               
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern                                                                                                                             
│   │   └── flower  # downloaded llff dataset                                                                                  
│   │   └── horns   # downloaded llff dataset
|   |   └── ...
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── ship    # downloaded synthetic dataset
|   |   └── ...
```

---

To train NeRF on different datasets: 

```
python run_nerf.py --config configs/{DATASET}.txt
```

replace `{DATASET}` with `trex` | `horns` | `flower` | `fortress` | `lego` | etc.

---

要在不同数据集上训练NeRF：

```
python run_nerf.py --config configs/{DATASET}.txt --render_only
```

将{DATASET}替换为 trex | horns | flower | fortress | lego 等。
