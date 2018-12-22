# TCPClip for VapourSynth
Python class for distributed video processing and encoding

## Usage

### Server side
```python
from TCPClip import TCPClipServer
<your vpy code>
TCPClipServer('<ip addr>', <port>, get_output(), <threads>)
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
from TCPClip import TCPClipClient
client = TCPClipClient('<ip addr>', <port>, <verbose>)
client.PipeToStdOut()
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
from TCPClip import TCPClipClient
from vapoursynth import core
clip = TCPClipClient('<ip addr>', <port>).Source(shutdown=True)
<your next vpy code>
clip.set_output()
```

#### Batches (VS Source mode)
```sh
vspipe -y EP01.vpy - | x264 ... --demuxer "y4m" --output "EP01.264" -
vspipe -y EP02.vpy - | x264 ... --demuxer "y4m" --output "EP02.264" -
...
vspipe -y EP12.vpy - | x264 ... --demuxer "y4m" --output "EP12.264" -
```
