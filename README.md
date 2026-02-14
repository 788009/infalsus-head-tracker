# In Falsus Head Tracker

基于头部旋转角度的 [In Falsus](https://infalsus.lowiro.com/) 天键解决方案，实际效果参考[这个视频](https://www.bilibili.com/video/BV1z4cqzWEQW/)。

## 使用方法

> [!NOTE]
>
> 由于程序只对我自己的设备进行了配置与优化，这里不提供二进制文件，你很可能需要直接修改代码内容（不仅仅是开头的配置）来获得最佳体验。

克隆本仓库并安装依赖。

```bash
git clone https://github.com/788009/infalsus-head-tracker.git
cd infalsus-head-tracker
pip install -r requirements.txt
```

> [!IMPORTANT]
> 
> 运行之前，建议打开 `headtracker.py` 查看开头的配置。

运行程序。

```bash
python headtracker.py
```

在游戏内，头部角度为 0 不一定一开始就映射到屏幕中心，此时可以用鼠标或触控板移动光标来校准。

遮住摄像头即可自由移动鼠标。

## 原理与踩坑

`headtracker.py` 是我与 Gemini 进行十几轮对话得到的结果，原理是用 [MediaPipe](https://github.com/google-ai-edge/mediapipe) 实时获取面部 landmarks，将鼻尖相对于双眼中心在画面上的 2D 偏移量映射到光标的横坐标，并使用 One Euro Filter 优化。

值得注意的一点是，用 `pyautogui` 来模拟鼠标是没有作用的，必须用 `pydirectinput`。

以下是尝试过的失败方案。

造轮子之前，先要尝试现有的轮子。我一共找到 2 个开源项目：
- [Enable Viacam](https://eviacam.crea-si.com/)
- [OpenTrack](https://github.com/opentrack/opentrack)

但效果都不理想。

之后尝试 PnP 算法，也不理想，抖动非常严重，于是换成现在的方案。

## 许可证

MIT License