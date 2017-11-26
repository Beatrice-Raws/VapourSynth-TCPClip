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

### Client side
```python
from TCPClip import TCPClipClient
client = TCPClipClient('<ip addr>', <port>, <verbose>)
client.PipeToStdOut()
```

#### Batches
```sh
py client.py | x264 ... --demuxer "y4m" --output "EP01.264" -
py client.py | x264 ... --demuxer "y4m" --output "EP02.264" -
...
py client.py | x264 ... --demuxer "y4m" --output "EP12.264" -
```
