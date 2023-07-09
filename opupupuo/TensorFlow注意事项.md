1、引用前需要先运行下述代码：
``` python
import numpy as np

np.object = object

np.bool = bool

np.int = int
```

2、设置gpu使用
``` python
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')

tf.config.set_visible_devices(physical_devices, 'GPU')

tf.config.set_visible_devices(physical_devices[0], 'GPU')

print("可用的物理GPU数量:", len(physical_devices))

print("当前可见的设备:", tf.config.get_visible_devices())
```