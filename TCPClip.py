# TCPClip Class by DJATOM
# Version 2.1
# License: MIT
# Why? Mainly for processing on server 1 and encoding on server 2, but it's also possible to distribute filtering chain.
#
# Usage:
#   Server side:
#       from TCPClip import TCPClipServer
#       <your vpy code>
#       TCPClipServer('<ip addr>', <port>, get_output())
#   Batches:
#       py EP01.py
#       py EP02.py
#       ...
#       py EP12.py
# 
#   Client side (plain encoding):
#       from TCPClip import TCPClipClient
#       client = TCPClipClient('<ip addr>', <port>)
#       client.pipe()
#   Batches:
#       py client.py | x264 ... --demuxer "y4m" --output "EP01.264" -
#       py client.py | x264 ... --demuxer "y4m" --output "EP02.264" -
#       ...
#       py client.py | x264 ... --demuxer "y4m" --output "EP12.264" -
#
#   Notice: only frame 0 props will affect Y4M header.
#
#   Client side (VS Source mode):
#       from TCPClip import TCPClipClient
#       from vapoursynth import core
#       clip = TCPClipClient('<ip addr>', <port>).source(shutdown=True)
#       <your next vpy code>
#       clip.set_output()
#   Batches:
#       vspipe -y EP01.vpy - | x264 ... --demuxer "y4m" --output "EP01.264" -
#       vspipe -y EP02.vpy - | x264 ... --demuxer "y4m" --output "EP02.264" -
#       ...
#       vspipe -y EP12.vpy - | x264 ... --demuxer "y4m" --output "EP12.264" -
#
#   Notice: frame properties will be also copied.
#   Notice No.2: If you're previewing your script, set shutdown=False. That will not call shutdown of TCPClipServer at the last frame.
#

from vapoursynth import core, VideoNode, VideoFrame
import numpy as np
import socket
import sys
import os
import time
import re
import pickle
import signal
import struct
import traceback
from threading import Thread
from enum import Enum
from typing import cast, Any, Union, List, Tuple

try:
    from psutil import Process
    def get_usable_cpus_count() -> int:
        return len(Process().cpu_affinity())
except:
    pass

def message(text: str = '') -> None:
    print(text, file=sys.stderr)

class Version(object):
    MAJOR   = 2
    MINOR   = 2

class Action(Enum):
    VERSION = 1
    CLOSE   = 2
    EXIT    = 3
    HEADER  = 4
    FRAME   = 5

class Helper():
    def __init__(self, soc: socket) -> None:
        self.soc = soc

    def send(self, msg: any) -> None:
        try:
            msg = struct.pack('>I', len(msg)) + msg
            self.soc.sendall(msg)
        except ConnectionResetError:
            message('Interrupted by client.')

    def recv(self) -> bytes:
        try:
            raw_msglen = self.recvall(4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]
            return self.recvall(msglen)
        except ConnectionResetError:
            message('Interrupted by client.')

    def recvall(self, n: int) -> bytes:
        data = b''
        try:
            while len(data) < n:
                packet = self.soc.recv(n - len(data))
                if not packet:
                    return None
                data += packet
        except ConnectionAbortedError:
            message('Connection aborted.')
        return data

class Server():
    def __init__(self, host: str = None, port: int = 14322, clip: VideoNode = None, threads: int = 0, verbose: bool = True) -> None:
        self.verbose = verbose 
        if not isinstance(clip, VideoNode):
            if self.verbose:
                message('Argument "clip" has wrong type.')
            sys.exit()
        self.clip = clip
        self.threads = core.num_threads if threads == 0 else threads
        self.frame_queue_buffer = dict()
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.verbose:
            message('Socket created.')
        try:
            self.soc.bind((host, port))
            if self.verbose:
                message('Socket bind complete.')
        except socket.error as msg:
            if self.verbose:
                message(f'Bind failed. Error: {sys.exc_info()}')
            sys.exit(2)
        self.soc.listen(2)
        if self.verbose:
            message('Listening the socket.')
        while True:
            self.conn, addr = self.soc.accept()
            ip, port = str(addr[0]), str(addr[1])
            if self.verbose:
                message(f'Accepting connection from {ip}:{port}.')
            try:
                Thread(target=self.server_loop, args=(ip, port)).start()
            except:
                message("Can't start main server loop!")
                traceback.print_exc()
        self.soc.close()

    def server_loop(self, ip: str, port: int) -> None:
        self.helper = Helper(self.conn)
        while True:
            input = self.helper.recv()
            try:
                query = pickle.loads(input)
            except:
                query = dict(type=Action.CLOSE)
            query_type = query['type']
            if query_type == Action.VERSION:
                if self.verbose:
                    message(f'Requested TCPClip version.')
                self.helper.send(pickle.dumps((Version.MAJOR, Version.MINOR)))
                if self.verbose:
                    message(f'TCPClip version sent.')
            elif query_type == Action.CLOSE:
                self.helper.send(pickle.dumps('close'))
                self.conn.close()
                if self.verbose:
                    message(f'Connection {ip}:{port} closed.')
                return
            elif query_type == Action.EXIT:
                self.helper.send(pickle.dumps('exit'))
                self.conn.close()
                if self.verbose:
                    message(f'Connection {ip}:{port} closed. Exiting, as client asked.')
                os._exit(0)
                return
            elif query_type == Action.HEADER:
                if self.verbose:
                    message(f'Requested clip info header.')
                self.get_meta()
                if self.verbose:
                    message(f'Clip info header sent.')
            elif query_type == Action.FRAME:
                if self.verbose:
                    message(f'Requested frame # {query["frame"]}.')
                self.get_frame(query['frame'], query['pipe'])
                if self.verbose:
                    message(f'Frame # {query["frame"]} sent.')
            else:
                self.conn.close()
                if self.verbose:
                    message(f'Received query has unknown type. Connection {ip}:{port} closed.')
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
            check_affinity = get_usable_cpus_count()
            usable_requests = min(self.threads, get_usable_cpus_count())
        except:
            usable_requests = self.threads
        for pf in range(cast(int, min(usable_requests, self.clip.num_frames - frame))):
            frame_to_pf = int(frame + pf)
            if frame_to_pf not in self.frame_queue_buffer:
                self.frame_queue_buffer[frame_to_pf] = self.clip.get_frame_async(frame_to_pf)
        out_frame = self.frame_queue_buffer.pop(frame).result()
        frame_data = list()
        for plane in range(self.clip.format.num_planes):
            plane_data = out_frame.get_read_array(plane)
            if pipe:
                out_plane = bytes(plane_data)
            else:
                out_plane = bytearray(plane_data)
            frame_data.append(out_plane)
        if not pipe:
            frame_props = dict(out_frame.props)
            self.helper.send(pickle.dumps((frame_data, frame_props)))
        else:
            self.helper.send(pickle.dumps(frame_data))

class Client():
    def __init__(self, host: str, port: int, verbose: bool = False) -> None:
        self.verbose = verbose
        self._stop = False # workaround for early interrupt
        try:
            self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.soc.connect((host, port))
            self.helper = Helper(self.soc)
        except ConnectionRefusedError:
            if self.verbose:
                message('Connection time-out reached. Probably closed port or server is down.')
            sys.exit(2)

    def query(self, data: dict) -> Any:
        try:
            self.helper.send(pickle.dumps(data))
            return pickle.loads(self.helper.recv())
        except:
            if self.verbose:
                message(f'Failed to make query {data}.')
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
            if self.verbose:
                message('Only 8-16 bit YUV or Gray formats are supported for Y4M outputs.')
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
        if self.verbose:
            start = time.perf_counter()
        server_version = self.version()
        if server_version != Version.MAJOR:
            if self.verbose:
                message(f'Version mismatch!\nServer: {server_version} | Client: {Version.MAJOR}')
            self.exit(2)
        header_info = self.get_meta()
        if len(header_info) == 0:
            if self.verbose:
                message('Wrong header info.')
            self.exit(2)
        if 'format' in header_info:
            clip_format = header_info['format']
        else:
            if self.verbose:
                message('Missing "Format".')
            self.exit(2)
        if 'props' in header_info:
            props = header_info['props']
        else:
            if self.verbose:
                message('Missing "props".')
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
        header = f'YUV4MPEG2 W{width} H{height} F{fps_num}:{fps_den} I{frameType} A{sar_num}:{sar_den} {csp} XYSCSS={csp} XLENGTH={num_frames}\n'
        sys.stdout.buffer.write(bytes(header, 'UTF-8'))
        signal.signal(signal.SIGINT, self.sigint_handler)
        for frame_number in range(num_frames):
            if self._stop:
                break
            if self.verbose:
                frameTime = time.perf_counter()
                eta = (frameTime - start) * (num_frames - (frame_number+1)) / ((frame_number+1))
            frame_data = self.get_frame(frame_number, pipe=True)
            sys.stdout.buffer.write(bytes('FRAME\n', 'UTF-8'))
            for plane in frame_data:
                sys.stdout.buffer.write(plane)
            if self.verbose:
                sys.stderr.write(f'Processing {frame_number}/{num_frames} ({frame_number/frameTime:.003f} fps) [{float(100 * frame_number / num_frames):.1f} %] [ETA: {int(eta//3600):d}:{int((eta//60)%60):02d}:{int(eta%60):02d}]  \r')
        self.exit()

    def as_source(self, shutdown: bool = False) -> VideoNode:
        def frame_copy(n: int, f: VideoFrame) -> VideoFrame:
            fout = f.copy()
            planes = fout.format.num_planes
            frame_data, frame_props = self.get_frame(n, pipe=False)
            dt = {1: np.uint8, 2: np.uint16, 4: np.float32}.get(fout.format.bytes_per_sample)
            for p in range(fout.format.num_planes):
                output_array = np.asarray(fout.get_write_array(p))
                if p == 0:
                    y, x = dummy.height, dummy.width
                else:
                    y, x = dummy.height >> dummy.format.subsampling_h, dummy.width >> dummy.format.subsampling_w
                output_array[:] = np.asarray(np.frombuffer(frame_data[p], dtype=dt)).reshape(y, x)
            for i in frame_props:
                fout.props[i] = frame_props[i]
            if shutdown and n == dummy.num_frames - 1:
                self.exit()
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