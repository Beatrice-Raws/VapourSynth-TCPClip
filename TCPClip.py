# TCPClip Class by DJATOM
# Version 2.3.0
# License: MIT
# Why? Mainly for processing on server 1 and encoding on server 2, but it's also possible to distribute filtering chain.
#
# Usage:
#   Server side:
#       from TCPClip import Server
#       <your vpy code>
#       Server('<ip addr>', <port>, get_output(), <threads>, <verbose>)
#   Batches:
#       py EP01.py
#       py EP02.py
#       ...
#       py EP12.py
# 
#   Client side (plain encoding):
#       from TCPClip import Client
#       client = Client('<ip addr>', <port>, <verbose>)
#       client.to_stdout()
#   Batches:
#       py client.py | x264 ... --demuxer "y4m" --output "EP01.264" -
#       py client.py | x264 ... --demuxer "y4m" --output "EP02.264" -
#       ...
#       py client.py | x264 ... --demuxer "y4m" --output "EP12.264" -
#
#   Notice: only frame 0 props will affect Y4M header.
#
#   Client side (VS Source mode):
#       from TCPClip import Client
#       from vapoursynth import core
#       clip = Client('<ip addr>', <port>, <verbose>).as_source(shutdown=True)
#       <your next vpy code>
#       clip.set_output()
#   Batches:
#       vspipe -y EP01.vpy - | x264 ... --demuxer "y4m" --output "EP01.264" -
#       vspipe -y EP02.vpy - | x264 ... --demuxer "y4m" --output "EP02.264" -
#       ...
#       vspipe -y EP12.vpy - | x264 ... --demuxer "y4m" --output "EP12.264" -
#
#   Notice: frame properties will be also copied.
#   Notice No.2: If you're previewing your script, set shutdown=False. That will not call shutdown of Server at the last frame.
#

from vapoursynth import core, VideoNode, VideoFrame # pylint: disable=no-name-in-module
import numpy as np
import socket
import sys
import os
import time
import re
import pickle
import signal
import struct
from threading import Thread
from enum import Enum, IntEnum
from typing import cast, Any, Union, List, Tuple

try:
    from psutil import Process
    def get_usable_cpus_count() -> int:
        return len(Process().cpu_affinity())
except:
    pass

class Version(object):
    MAJOR   = 2
    MINOR   = 2
    BUGFIX  = 2

class Action(Enum):
    VERSION = 1
    CLOSE   = 2
    EXIT    = 3
    HEADER  = 4
    FRAME   = 5

class LL(IntEnum):
    Crit    = 1
    Warn    = 2
    Info    = 3
    Debug   = 4

class Util(object):
    def __new__(cls):
        """ Instantiate Utils as Singleton object """
        if not hasattr(cls, 'instance'):
            cls.instance = super(Util, cls).__new__(cls)
        return cls.instance

    def get_caller(self) -> Union[str, tuple]:
        def stack_(frame):
            framelist = []
            while frame:
                framelist.append(frame)
                frame = frame.f_back
            return framelist
        stack = stack_(sys._getframe(1))
        if len(stack) < 2:
            return 'Main'
        parentframe = stack[1]
        if 'self' in parentframe.f_locals:
            parrent_cls = parentframe.f_locals['self']
            return (parrent_cls.__class__.__name__, parrent_cls.log_level)
        return 'Main'

    def as_enum(self, level: str = 'info') -> LL:
        if isinstance(level, str):
            level = {'crit': LL.Crit, 'warn': LL.Warn, 'info': LL.Info, 'debug': LL.Debug}.get(level)
        return level

    def message(self, level: str, text: str) -> None:
        facility, parrent_level = self.get_caller()
        if self.as_enum(parrent_level) >= Util().as_enum(level):
            print(f'{facility:6s} [{level}]: {text}', file=sys.stderr)

class Helper():
    def __init__(self, soc: socket, log_level: Union[str, LL] = None) -> None:
        self.soc = soc
        self.log_level = Util().as_enum(log_level) if isinstance(log_level, str) else log_level

    def send(self, msg: any) -> None:
        try:
            msg = struct.pack('>I', len(msg)) + msg
            self.soc.sendall(msg)
        except ConnectionResetError:
            Util().message('crit', 'send - interrupted by client.')

    def recv(self) -> bytes:
        try:
            raw_msglen = self.recvall(4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]
            return self.recvall(msglen)
        except ConnectionResetError:
            Util().message('crit', 'recv - interrupted by client.')

    def recvall(self, n: int) -> bytes:
        data = b''
        try:
            while len(data) < n:
                packet = self.soc.recv(n - len(data))
                if not packet:
                    return None
                data += packet
        except ConnectionAbortedError:
            Util().message('crit', 'recvall - connection aborted.')
        return data

class Server():
    def __init__(self, host: str = None, port: int = 14322, clip: VideoNode = None, threads: int = 0, log_level: Union[str, LL] = 'info') -> None:
        self.log_level = Util().as_enum(log_level) if isinstance(log_level, str) else log_level
        if not isinstance(clip, VideoNode):
            Util().message('crit', 'argument "clip" has wrong type.')
            sys.exit(2)
        self.clip = clip
        self.threads = core.num_threads if threads == 0 else threads
        self.frame_queue_buffer = dict()
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        Util().message('info', 'socket created.')
        try:
            self.soc.bind((host, port))
            Util().message('info', 'socket bind complete.')
        except socket.error:
            Util().message('crit', f'bind failed. Error: {sys.exc_info()}')
            sys.exit(2)
        self.soc.listen(2)
        Util().message('info', 'listening the socket.')
        while True:
                self.conn, addr = self.soc.accept()
                ip, port = str(addr[0]), str(addr[1])
                Util().message('info', f'accepting connection from {ip}:{port}.')
                try:
                    Thread(target=self.server_loop, args=(ip, port)).start()
                except:
                    Util().message('crit', f'can\'t start main server loop! {sys.exc_info()}')
        self.soc.close()

    def server_loop(self, ip: str, port: int) -> None:
        self.helper = Helper(self.conn, self.log_level)
        while True:
            input = self.helper.recv()
            try:
                query = pickle.loads(input)
            except:
                query = dict(type=Action.CLOSE)
            query_type = query['type']
            if query_type == Action.VERSION:
                Util().message('debug', f'requested TCPClip version.')
                self.helper.send(pickle.dumps((Version.MAJOR, Version.MINOR, Version.BUGFIX)))
                Util().message('debug', f'TCPClip version sent.')
            elif query_type == Action.CLOSE:
                self.helper.send(pickle.dumps('close'))
                for frame in list(self.frame_queue_buffer):
                    del self.frame_queue_buffer[frame]
                    Util().message('debug', f'frame {frame} freed.')
                self.frame_queue_buffer.clear()
                self.conn.close()
                Util().message('info', f'connection {ip}:{port} closed.')
                return
            elif query_type == Action.EXIT:
                self.helper.send(pickle.dumps('exit'))
                self.conn.close()
                Util().message('info', f'connection {ip}:{port} closed. Exiting, as client asked.')
                os._exit(0)
                return
            elif query_type == Action.HEADER:
                Util().message('debug', f'requested clip info header.')
                self.get_meta()
                Util().message('debug', f'clip info header sent.')
            elif query_type == Action.FRAME:
                Util().message('debug', f'requested frame # {query["frame"]}.')
                self.get_frame(query['frame'], query['pipe'])
                Util().message('debug', f'frame # {query["frame"]} sent.')
            else:
                self.conn.close()
                Util().message('warn', f'received query has unknown type. Connection {ip}:{port} closed.')
                return

    def get_meta(self) -> None:
        clip = self.clip
        props = dict(clip.get_frame(0).props)
        self.helper.send(
            pickle.dumps(
                dict(
                    format = dict(
                        id = clip.format.id, 
                        name = clip.format.name,
                        color_family = int(clip.format.color_family), 
                        sample_type = int(clip.format.sample_type), 
                        bits_per_sample = clip.format.bits_per_sample, 
                        bytes_per_sample = clip.format.bytes_per_sample, 
                        subsampling_w = clip.format.subsampling_w, 
                        subsampling_h = clip.format.subsampling_h, 
                        num_planes = clip.format.num_planes
                    ), 
                    width = clip.width, 
                    height = clip.height, 
                    num_frames = clip.num_frames, 
                    fps_numerator = clip.fps.numerator, 
                    fps_denominator = clip.fps.denominator,
                    props = props
                )
            )
        )

    def get_frame(self, frame: int = 0, pipe: bool = False) -> None:
        try:
            usable_requests = min(self.threads, get_usable_cpus_count())
        except:
            usable_requests = self.threads
        for pf in range(min(usable_requests, self.clip.num_frames - frame)):
            frame_to_pf = int(frame + pf)
            if frame_to_pf not in self.frame_queue_buffer:
                self.frame_queue_buffer[frame_to_pf] = self.clip.get_frame_async(frame_to_pf)
                Util().message('debug', f'get_frame_async({frame_to_pf}) called at get_frame({frame}).')
        out_frame = self.frame_queue_buffer.pop(frame).result()
        frame_data = [np.asarray(plane) for plane in out_frame.planes()]
        if not pipe:
            frame_props = dict(out_frame.props)
            self.helper.send(pickle.dumps((frame_data, frame_props)))
        else:
            self.helper.send(pickle.dumps(frame_data))
        del out_frame

class Client():
    def __init__(self, host: str, port: int = 14322, log_level: str = 'info', shutdown: bool = False) -> None:
        self.log_level = Util().as_enum(log_level) if isinstance(log_level, str) else log_level
        self.shutdown = shutdown
        self._stop = False # workaround for early interrupt
        try:
            self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.soc.connect((host, port))
            self.helper = Helper(self.soc, self.log_level)
        except ConnectionRefusedError:
            Util().message('crit', 'connection time-out reached. Probably closed port or server is down.')
            sys.exit(2)

    def __del__(self) -> None:
        if self.shutdown: # kill server on exit
            self.exit()

    def query(self, data: dict) -> Any:
        try:
            self.helper.send(pickle.dumps(data))
            return pickle.loads(self.helper.recv())
        except:
            Util().message('crit', f'failed to make query {data}.')
            sys.exit(2)

    def version(self, minor: bool = False) -> Union[tuple, int]:
        v = self.query(dict(type=Action.VERSION))
        if minor:
            return v
        else:
            return v[0]

    def close(self) -> None:
        self.query(dict(type=Action.CLOSE))
        self.soc.close()

    def exit(self, code: int = 0) -> None:
        self.query(dict(type=Action.EXIT))
        self.soc.close()
        sys.exit(code)

    def get_meta(self) -> dict:
        return self.query(dict(type=Action.HEADER))

    def get_frame(self, frame: int, pipe: bool = False) -> Union[Tuple[list, dict], list]:
        return self.query(dict(type=Action.FRAME, frame=frame, pipe=pipe))

    def get_y4m_csp(self, clip_format: dict) -> str:
        if clip_format['bits_per_sample'] > 16:
            Util().message('crit', 'only 8-16 bit YUV or Gray formats are supported for Y4M outputs.')
            self.exit(2)
        bits = clip_format['bits_per_sample']
        if clip_format['num_planes'] == 3:
            y = 4
            w = y >> clip_format['subsampling_w']
            h = y >> clip_format['subsampling_h']
            u = abs(w)
            v = abs(y - w - h)
            csp = f'{y}{u}{v}'
        else:
            csp = None
        return {1: f'Cmono{bits}', 3: f'C{csp}p{bits}'}.get(clip_format['num_planes'], 'C420p8')

    def sigint_handler(self, *args) -> None:
        self._stop = True

    def to_stdout(self) -> None:
        if self.log_level >= LL.Info:
            start = time.perf_counter()
        server_version = self.version()
        if server_version != Version.MAJOR:
            Util().message('crit', f'version mismatch!\nServer: {server_version} | Client: {Version.MAJOR}')
            self.exit(2)
        header_info = self.get_meta()
        if len(header_info) == 0:
            Util().message('crit', 'wrong header info.')
            self.exit(2)
        if 'format' in header_info:
            clip_format = header_info['format']
        else:
            Util().message('crit', 'missing "Format".')
            self.exit(2)
        if 'props' in header_info:
            props = header_info['props']
        else:
            Util().message('crit', 'missing "props".')
            self.exit(2)
        if '_FieldBased' in props:
            frameType = {2: 't', 1: 'b', 0: 'p'}.get(props['_FieldBased'], 'p')
        else:
            frameType = 'p'
        if '_SARNum' and '_SARDen' in props:
            sar_num, sar_den = props['_SARNum'], props['_SARDen']
        else:
            sar_num, sar_den = 0, 0
        num_frames = header_info['num_frames']
        width = header_info['width']
        height = header_info['height']
        fps_num = header_info['fps_numerator']
        fps_den = header_info['fps_denominator']
        csp = self.get_y4m_csp(clip_format)
        sys.stdout.buffer.write(bytes(f'YUV4MPEG2 W{width} H{height} F{fps_num}:{fps_den} I{frameType} A{sar_num}:{sar_den} {csp} XYSCSS={csp} XLENGTH={num_frames}\n', 'UTF-8'))
        signal.signal(signal.SIGINT, self.sigint_handler)
        for frame_number in range(num_frames):
            if self._stop:
                break
            if self.log_level >= LL.Info:
                frameTime = time.perf_counter()
                eta = (frameTime - start) * (num_frames - (frame_number+1)) / ((frame_number+1))
            frame_data = self.get_frame(frame_number, pipe=True)
            sys.stdout.buffer.write(bytes('FRAME\n', 'UTF-8'))
            for plane in frame_data:
                sys.stdout.buffer.write(plane)
            if self.log_level >= LL.Info:
                sys.stderr.write(f'Processing {frame_number}/{num_frames} ({frame_number/frameTime:.003f} fps) [{float(100 * frame_number / num_frames):.1f} %] [ETA: {int(eta//3600):d}:{int((eta//60)%60):02d}:{int(eta%60):02d}]  \r')

    def as_source(self) -> VideoNode:
        def frame_copy(n: int, f: VideoFrame) -> VideoFrame:
            fout = f.copy()
            frame_data, frame_props = self.get_frame(n, pipe=False)
            for p in range(fout.format.num_planes):
                np.asarray(fout.get_write_array(p))[:] = frame_data[p]
            for i in frame_props:
                fout.props[i] = frame_props[i]
            return fout
        server_version = self.version()
        assert server_version == Version.MAJOR, f'Version mismatch!\nServer: {server_version} | Client: {Version.MAJOR}'
        header_info = self.get_meta()
        assert len(header_info) > 0, 'Wrong header info.'
        assert 'format' in header_info, 'Missing "Format".'
        clip_format  = header_info['format']
        source_format = core.register_format(clip_format['color_family'], clip_format['sample_type'], clip_format['bits_per_sample'], clip_format['subsampling_w'], clip_format['subsampling_h'])
        dummy = core.std.BlankClip(width=header_info['width'], height=header_info['height'], format=source_format, length=header_info['num_frames'], fpsnum=header_info['fps_numerator'], fpsden=header_info['fps_denominator'], keep=True)
        source = core.std.ModifyFrame(dummy, dummy, frame_copy)
        return source