# TCPClip Class by DJATOM
# Version 0.2
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
#       client.PipeToStdOut()
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
#       clip = TCPClipClient('<ip addr>', <port>).Source(shutdown=True)
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

from vapoursynth import core, SampleType
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
import traceback
from collections import namedtuple
from enum import Enum

TCPClipVersionMAJOR, TCPClipVersionMINOR = 2, 1

class TCPClipAction(Enum):
    VERSION = 1
    CLOSE = 2
    EXIT = 3
    HEADER = 4
    FRAME = 5

class TCPClipHelper():
    def __init__(self, soc):
        self.soc = soc

    def send(self, msg):
        try:
            msg = struct.pack('>I', len(msg)) + msg
            self.soc.sendall(msg)
        except ConnectionResetError:
            print('Interrupted by Client.', file=sys.stderr)

    def recv(self):
        try:
            raw_msglen = self.recvall(4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]
            return self.recvall(msglen)
        except ConnectionResetError:
            print('Interrupted by Client.', file=sys.stderr)

    def recvall(self, n):
        data = b''
        try:
            while len(data) < n:
                packet = self.soc.recv(n - len(data))
                if not packet:
                    return None
                data += packet
        except ConnectionAbortedError:
            print('Connection Aborted.', file=sys.stderr)
        return data

class TCPClipServer():
    def __init__(self, host, port, clip, threads=0):
        self.clip = clip
        self.threads = os.cpu_count() if threads == 0 else threads
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print('Socket created.', file=sys.stderr)
        try:
            self.soc.bind((host, port))
            print('Socket bind complete.', file=sys.stderr)
        except socket.error as msg:
            print(f'Bind failed. Error: {sys.exc_info()}')
            sys.exit()
        self.soc.listen(2)
        print('Listening the socket.', file=sys.stderr)
        while True:
            self.conn, addr = self.soc.accept()
            ip, port = str(addr[0]), str(addr[1])
            print(f'Accepting connection from {ip}:{port}.', file=sys.stderr)
            try:
                Thread(target=self.ServerLoop, args=(ip, port)).start()
            except:
                print("Can't start main server loop!", file=sys.stderr)
                traceback.print_exc()
        self.soc.close()

    def ServerLoop(self, ip, port):
        self.helper = TCPClipHelper(self.conn)
        while True:
            input = self.helper.recv()
            try:
                query = pickle.loads(input)
            except:
                query = dict(type=TCPClipAction.CLOSE)
            qType = query['type']
            if qType == TCPClipAction.VERSION:
                self.helper.send(pickle.dumps((TCPClipVersionMAJOR, TCPClipVersionMINOR)))
            elif qType == TCPClipAction.CLOSE:
                self.helper.send(pickle.dumps('close'))
                self.conn.close()
                print(f'Connection {ip}:{port} closed.', file=sys.stderr)
                return
            elif qType == TCPClipAction.EXIT:
                self.helper.send(pickle.dumps('exit'))
                self.conn.close()
                print(f'Connection {ip}:{port} closed. Exiting, as client asked.', file=sys.stderr)
                os._exit(0)
                return
            elif qType == TCPClipAction.HEADER:
                self.GetMeta()
            elif qType == TCPClipAction.FRAME:
                self.GetFrame(query['frame'], query['pipe'])
            else:
                self.conn.close()
                print(f'Received query has unknown type. Connection {ip}:{port} closed.', file=sys.stderr)
                return

    def GetMeta(self):
        clip = self.clip
        props = clip.get_frame(0).props
        frameprops = dict()
        for prop in props:
            frameprops[prop] = props[prop]
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
                    props = frameprops
                )
            )
        )

    def GetFrame(self, frame, pipe=False):
        if self.threads > 1:
            for pf in range(self.threads):
                if pf < self.clip.num_frames:
                    self.clip.get_frame_async(frame+pf)
        singleframe = self.clip.get_frame(frame)
        data = []
        for i in range(self.clip.format.num_planes):
            planedata = singleframe.get_read_array(i)
            if pipe:
                outplane = bytes(planedata)
            else:
                outplane = bytearray(planedata)
            data.append(outplane)
        frameprops = dict()
        if not pipe:
            props = singleframe.props
            for prop in props:
                frameprops[prop] = props[prop]
        self.helper.send(pickle.dumps((data, frameprops)))

class TCPClipClient():
    def __init__(self, host, port, verbose=False):
        self.verbose = verbose
        try:
            self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.soc.connect((host, port))
        except ConnectionRefusedError:
            print('Connection time-out reached. Probably closed port or server is down.', file=sys.stderr)
            sys.exit(2)

    def Query(self, data):
        try:
            self.helper = TCPClipHelper(self.soc)
            self.helper.send(pickle.dumps(data))
            result = self.helper.recv()
            return pickle.loads(result)
        except:
            print('Failed to make Query.', file=sys.stderr)
            sys.exit(2)

    def Version(self):
        return self.Query(dict(type = TCPClipAction.VERSION))

    def Close(self):
        self.Query(dict(type = TCPClipAction.CLOSE))
        self.soc.close()

    def Exit(self):
        return self.Query(dict(type = TCPClipAction.EXIT))

    def GetMeta(self):
        return self.Query(dict(type = TCPClipAction.HEADER))

    def GetFrame(self, number, pipe = False):
        return self.Query(dict(type = TCPClipAction.FRAME, frame = number, pipe = pipe))

    def GetCSP(self, name, num_planes):
        if num_planes == 3:
            formatdata = re.findall('YUV(\d+)P(\d+)', name, flags=re.IGNORECASE)
            if len(formatdata) < 1:
                print('Received frame has 3 planes, but they are in non-YUV format. Not supported.', file=sys.stderr)
                self.Exit()
                sys.exit(2)
            csp, bits = formatdata[0]
            return f'C{csp}p{bits}'
        elif num_planes == 1:
            formatdata = re.findall('GRAY(\d+)', name, flags=re.IGNORECASE)
            if len(formatdata) < 1:
                print('Received frame has 1 plane, but it is in non-GRAY format. Not supported.', file=sys.stderr)
                self.Exit()
                sys.exit(2)
            bits = formatdata[0]
            return f'Cmono{bits}'
        else:
            print(f'Received frame has {num_planes} planes. Not supported.', file=sys.stderr)
            self.Exit()
            sys.exit(2)

    def SigIntHandler(self, *args):
        self.Exit()
        sys.exit(1)

    def PipeToStdOut(self):
        start = time.perf_counter()
        ServerVersionMajor, _ = self.Version()
        if ServerVersionMajor != TCPClipVersionMAJOR:
            print(f'Version mismatch!\nServer: {ServerVersionMajor} | Client: {TCPClipVersionMAJOR}', file=sys.stderr)
            sys.exit(2)
        hInfo = self.GetMeta()
        if len(hInfo) == 0:
            print('Wrong header info.', file=sys.stderr)
            self.Exit()
            sys.exit(2)
        if 'format' in hInfo:
            cFormat = hInfo['format']
        else:
            print('Missing "Format".', file=sys.stderr)
            self.Exit()
            sys.exit(2)
        if 'props' in hInfo:
            props = hInfo['props']
        else:
            print('Missing "props".', file=sys.stderr)
            self.Exit()
            sys.exit(2)
        if '_FieldBased' in props:
            frameType = {2: 't', 1: 'b', 0: 'p'}.get(props['_FieldBased'], 'p')
        else:
            frameType = 'p'
        if '_SARNum' and '_SARDen' in props:
            sarNum, sarDen = props['_SARNum'], props['_SARDen']
        else:
            sarNum, sarDen = 0, 0
        num_frames = hInfo['num_frames']
        width = hInfo['width']
        height = hInfo['height']
        fps_num = hInfo['fps_numerator']
        fps_den = hInfo['fps_denominator']
        csp = self.GetCSP(cFormat['name'], cFormat['num_planes']),
        header = f'YUV4MPEG2 W{width} H{height} F{fps_num}:{fps_den} I{frameType} A{sarNum}:{sarDen} {csp} XYSCSS={csp} XLENGTH={num_frames}\n'
        sys.stdout.buffer.write(bytes(header, 'UTF-8'))
        signal.signal(signal.SIGINT, self.SigIntHandler)
        for frameNumber in range(num_frames):
            if self.verbose:
                frameTime = time.perf_counter()
                eta = (frameTime - start) * (num_frames - (frameNumber+1)) / ((frameNumber+1))
            frameData, _ = self.GetFrame(frameNumber, pipe=True)
            sys.stdout.buffer.write(bytes('FRAME\n', 'UTF-8'))
            for plane in frameData:
                sys.stdout.buffer.write(plane)
            if self.verbose:
                sys.stderr.write('Processing {}/{} ({:.003f} fps) [{:.1f} %] [ETA: {:d}:{:02d}:{:02d}]  \r'.format(frameNumber, num_frames, frameNumber/frameTime, float(100 * frameNumber / num_frames), int(eta//3600), int((eta//60)%60), int(eta%60)))
        self.Exit()

    def Source(self, shutdown=False):
        def frameCopy(n, f):
            fout = f.copy()
            planes = fout.format.num_planes
            frameData, frameProps = self.GetFrame(n, pipe=False)
            dt = {1: np.uint8, 2: np.uint16, 4: np.float32}.get(fout.format.bytes_per_sample)
            for p in range(fout.format.num_planes):
                output_array = np.asarray(fout.get_write_array(p))
                if p == 0:
                    y, x = dummy.height, dummy.width
                else:
                    y, x = dummy.height >> dummy.format.subsampling_h, dummy.width >> dummy.format.subsampling_w
                output_array[:] = np.asarray(np.frombuffer(frameData[p], dtype=dt)).reshape(y, x)
            for i in frameProps:
                fout.props[i] = frameProps[i]
            if shutdown and n == dummy.num_frames - 1:
                self.Exit()
            return fout
        ServerVersionMajor, _ = self.Version()
        assert ServerVersionMajor == TCPClipVersionMAJOR, f'Version mismatch!\nServer: {ServerVersionMajor} | Client: {TCPClipVersionMAJOR}'
        hInfo = self.GetMeta()
        assert len(hInfo) > 0, 'Wrong header info.'
        assert 'format' in hInfo, 'Missing "Format".'
        cFmt  = hInfo['format']
        srcFormat = core.register_format(cFmt['color_family'],cFmt['sample_type'],cFmt['bits_per_sample'],cFmt['subsampling_w'],cFmt['subsampling_h'])
        dummy = core.std.BlankClip(width=hInfo['width'], height=hInfo['height'], format=srcFormat, length=hInfo['num_frames'], fpsnum=hInfo['fps_numerator'], fpsden=hInfo['fps_denominator'])
        source = core.std.ModifyFrame(dummy, dummy, frameCopy)
        return source