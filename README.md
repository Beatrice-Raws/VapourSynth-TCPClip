# TCPClip for VapourSynth
Python class for distributed video processing and encoding

## Usage

### Server side (adjust threads or don't set varibale for auto-detection)
```python
from TCPClip import Server
<your vpy code>
Server('<ip addr>', 14322, get_output(), threads=8, log_level='info', compression_method=None, compression_level=1, compression_threads=1)
```

#### Batches
```sh
py EP01.py
py EP02.py
...
py EP12.py
```

### Client side (plain encoding)
```python
from TCPClip import Client
Client('<ip addr>', port=14322, log_level='info', shutdown=True).to_stdout()
```

#### Batches (plain encoding)
```sh
py client.py | x264 ... --demuxer "y4m" --output "EP01.264" -
py client.py | x264 ... --demuxer "y4m" --output "EP02.264" -
...
py client.py | x264 ... --demuxer "y4m" --output "EP12.264" -
```

### Client side (VS Source mode)
```python
from TCPClip import Client
from vapoursynth import core
clip = Client('<ip addr>', port=14322, log_level='info', shutdown=True).as_source()
<your extra vpy code>
clip.set_output()
```

#### Batches (VS Source mode)
```sh
vspipe -y EP01.vpy - | x264 ... --demuxer "y4m" --output "EP01.264" -
vspipe -y EP02.vpy - | x264 ... --demuxer "y4m" --output "EP02.264" -
...
vspipe -y EP12.vpy - | x264 ... --demuxer "y4m" --output "EP12.264" -
```
