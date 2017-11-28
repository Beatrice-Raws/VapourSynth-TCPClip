# TCPClip Class by DJATOM
# Version 0.1
# License: MIT
# Why? Mainly for processing on server 1 and encoding on server 2.
# I wanted to write a replacement for TCPDeliver, but I dunno how to re-create VideoNode/VideoFrame with pure Python. 
# TCPSource concept is better when we want to distribute filtering chain.
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
#   Client side:
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

from vapoursynth import core, VideoNode, VideoFrame # not really used, but I might use them in future
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
from enum import IntEnum

class TCPClipVersion(IntEnum):
    Major = 0
    Minor = 1

class TCPClipAction(IntEnum):
    Version = 1
    Close = 2
    Exit = 3
    Header = 4
    Frame = 5

class TCPClipHelper():
    def __init__(self, soc):
        self.soc = soc

    def send(self, msg):
        try:
            msg = struct.pack('>I', len(msg)) + msg
            self.soc.sendall(msg)

        except ConnectionResetError:
            print ('Interrupted by Client.')


    def recv(self):
        try:
            raw_msglen = self.recvall(4)
            if not raw_msglen:
                return None

            msglen = struct.unpack('>I', raw_msglen)[0]

            return self.recvall(msglen)

        except ConnectionResetError:
            print ('Interrupted by Client.')

    def recvall(self, n):
        data = b''

        try:
            while len(data) < n:
                packet = self.soc.recv(n - len(data))

                if not packet:
                    return None

                data += packet

        except ConnectionAbortedError:
            print ('Connection Aborted.')

        return data

class TCPClipServer():
    def __init__(self, host, port, clip, threads=0):
        self.clip = clip
        self.threads = os.cpu_count() if threads == 0 else threads
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print ('Socket created.')

        try:
            self.soc.bind((host, port))
            print ('Socket bind complete.')

        except socket.error as msg:
            print ('Bind failed. Error: ' + str(sys.exc_info()))
            sys.exit()

        self.soc.listen(2)
        print ('Listening the socket.')

        while True:
            self.conn, addr = self.soc.accept()
            ip, port = str(addr[0]), str(addr[1])
            print ('Accepting connection from {}:{}.'.format(ip, port))

            try:
                Thread(target=self.ServerLoop, args=(ip, port)).start()

            except:
                print ('Terrible error!')
                traceback.print_exc()

        self.soc.close()

    def ServerLoop(self, ip, port):
        self.helper = TCPClipHelper(self.conn)

        while True:
            input = self.helper.recv()

            try:
                query = pickle.loads(input)
            except:
                query = dict(type = TCPClipAction.Close)

            qtype = query['type']
            if qtype == TCPClipAction.Version:
                self.helper.send(
                    pickle.dumps(
                        dict(
                            major = TCPClipVersion.Major, 
                            minor = TCPClipVersion.Minor
                        ), 
                        -1
                    )
                )
            elif qtype == TCPClipAction.Close:
                self.helper.send(
                    pickle.dumps('close', -1)
                )
                self.conn.close()
                print ('Connection {}:{} closed.'.format(ip, port))
                break
            elif qtype == TCPClipAction.Exit:
                self.helper.send(
                    pickle.dumps('exit', -1)
                )
                self.conn.close()
                print ('Connection {}:{} closed. Exiting, as client asked.'.format(ip, port))
                os._exit(0)
                break
            elif qtype == TCPClipAction.Header:
                self.GetMeta()
            elif qtype == TCPClipAction.Frame:
                self.GetFrame(query['frame'])
            else:
                self.conn.close()
                print ('Received query has unknown type. Connection {}:{} closed.'.format(ip, port))
                break

    def GetMeta(self):
        clip = self.clip

        props = self.clip.get_frame(0).props
        frameprops = []
        for prop in props:
            frameprops.append(prop)

        self.helper.send(
            pickle.dumps(
                dict(
                    format = dict(
                        id = clip.format.id, 
                        name = clip.format.name,
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
                ), 
                -1
            )
        )

    def GetFrame(self, frame):
        for pf in range(self.threads):
            self.clip.get_frame_async(frame+pf)
        singleframe = self.clip.get_frame(frame)

        data=[]
        for i in range(self.clip.format.num_planes):
            data.append( 
                self.GetPlane(singleframe, i)
            )

        self.helper.send(
            pickle.dumps(data, -1)
        )

    def GetPlane(self, singleframe, plane):
        return bytes(singleframe.get_read_array(plane))

class TCPClipClient():
    def __init__(self, host, port, verbose=False):
        self.verbose = verbose
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.connect((host, port))

    def Query(self, data):
        try:
            self.helper = TCPClipHelper(self.soc)
            self.helper.send(pickle.dumps(data, -1))
            result = self.helper.recv()

            return pickle.loads(result)
        except:
            print('Failed to make Query.')
            sys.exit(1)

    def Version(self):
        return self.Query(
            dict(
                type = TCPClipAction.Version
            )
        )

    def Close(self):
        self.Query(
            dict(
                type = TCPClipAction.Close
            )
        )
        self.soc.close()

    def Exit(self):
        return self.Query(
            dict(
                type = TCPClipAction.Exit
            )
        )

    def GetMeta(self):
        return self.Query(
            dict(
                type = TCPClipAction.Header
            )
        )

    def GetFrame(self, number):
        return self.Query(
            dict(
                type = TCPClipAction.Frame, 
                frame = number
            )
        )

    def GetCSP(self, name, num_planes): # need to write better validation on non-YUV formats, I don't support it
        if num_planes == 3:
            matched = re.findall('YUV(\d+)P(\d+)', name)
            if not matched:
                print('Received frame has 3 planes, but they are in non-YUV format. Not supported.')
            csp, bits = matched[0]
            return 'C{}p{}'.format(csp, bits)
        elif num_planes == 1:
            matched = re.findall('GRAY(\d+)', name)
            if not matched:
                print('Received frame has 1 plane, but it is in non-GRAY format. Not supported.')
            bits = matched[0]
            return 'Cmono{}'.format(bits)
        else:
            print('Received frame has {} planes. Not supported.'.format(num_planes))

    def SigIntHandler(self, *args):
        self.Exit()
        sys.exit(0)

    def PipeToStdOut(self):
        ServerVersion = self.Version()
        if ServerVersion['major'] != TCPClipVersion.Major:
            print('Version mismatch!\nServer: {} | Client: {}'.format(ServerVersion['major'], TCPClipVersion.Major))
            self.Exit()
            sys.exit(1)

        headerInfo = self.GetMeta()
        if len(headerInfo) == 0:
            print('Wrong header info.')
            self.Exit()
            sys.exit(1)

        if 'format' in headerInfo:
            cFormat = headerInfo['format']
        else:
            print('Missing "Format".')
            self.Exit()
            sys.exit(1)

        if 'props' in headerInfo:
            props = headerInfo['props']
        else:
            print('Missing "props".')
            self.Exit()
            sys.exit(1)

        if '_FieldBased' in props:
            frameType = {2: "t", 1: "b", 0: "p"}.get(props['_FieldBased'])
        else:
            frameType = 'p'

        if '_SARNum' and '_SARDen' in props:
            sarNum, sarDen = props['_SARNum'], props['_SARDen']
        else:
            sarNum, sarDen = 0, 0

        header = 'YUV4MPEG2 W{width} H{height} F{fps_num}:{fps_den} I{frame_type} A{sar_num}:{sar_den} {csp} XYSCSS={csp} XLENGTH={frames_num}\n'.format(
                width = headerInfo['width'], 
                height = headerInfo['height'], 
                fps_num = headerInfo['fps_numerator'], 
                fps_den = headerInfo['fps_denominator'], 
                frame_type = frameType, 
                sar_num = sarNum, 
                sar_den = sarDen, 
                csp = self.GetCSP(
                    cFormat['name'], 
                    cFormat['num_planes']
                ), 
                frames_num = headerInfo['num_frames']
            )
        sys.stdout.buffer.write(bytes(header, 'UTF-8'))

        signal.signal(signal.SIGINT, self.SigIntHandler)

        for frameNumber in range(headerInfo['num_frames']):
            frameData = self.GetFrame(frameNumber)

            sys.stdout.buffer.write(bytes('FRAME\n', 'UTF-8'))

            for plane in frameData:
                sys.stdout.buffer.write(plane)
            if self.verbose:
                sys.stderr.write('Current frame: {i} \r'.format(i=frameNumber))

        self.Exit()
