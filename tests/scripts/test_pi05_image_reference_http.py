import hashlib
from http.server import ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace
from urllib import request, error

import numpy as np
import pytest
import torch

from lerobot.scripts.pi05_array_wire import CONTENT_TYPE, dump_arrays
from lerobot.scripts.pi05_image_resources import ImageResources, SOURCES
from lerobot.scripts.pi05_image_inputs import ImageTensorCache
from lerobot.scripts.pi05_vla_server import VLARequestHandler, RolloutRecorder, decode_observation_packet, AsyncRTCPolicy
from lerobot.scripts.pi05_vla_server import _decode_jpeg_window, resize_with_pad_torch, _server_image_resources
from test_pi05_image_table import inputs


def test_http_ack_references_and_raw_worker_recording(tmp_path):
    _legacy, flat, _history = inputs()
    recorder = RolloutRecorder(str(tmp_path))
    packets=[]
    def submit(packet):
        packets.append(packet)
        recorder.observation(packet)
    engine=SimpleNamespace(input_spec=lambda:{"schema_id":str(flat['model_input_schema'].item())},
                           validate_input=lambda _packet:None,submit=submit)
    server=ThreadingHTTPServer(('127.0.0.1',0),VLARequestHandler)
    server.engine,server.max_payload_bytes=engine,32*1024*1024
    thread=Thread(target=server.serve_forever,daemon=True)
    thread.start()
    sender=ImageResources()
    try:
        for index in range(3):
            wire=sender.pack(flat)
            payload=dump_arrays(wire)
            digest=hashlib.sha256(payload).hexdigest()
            req=request.Request(f'http://127.0.0.1:{server.server_port}/v1/observations',
                data=payload,headers={'Content-Type':CONTENT_TYPE},method='POST')
            with request.urlopen(req,timeout=3) as response:
                accepted=json.load(response)
            assert accepted['image_input_sha256']==digest
            sender.commit(wire,digest)
            assert (tmp_path/'vla/inputs'/(digest+'.awb')).read_bytes()==payload
            assert (SOURCES in wire)==(index>0)
        assert all(packet.decoded_images is not None and packet.base_rgb is not None for packet in packets)
        for packet in packets[1:]:
            for key,value in packets[0].model_image_history.data.items():
                np.testing.assert_array_equal(packet.model_image_history.data[key],value)
        bad=dict(wire,control_epoch=np.int64(int(flat['control_epoch'])+1))
        req=request.Request(f'http://127.0.0.1:{server.server_port}/v1/observations',
            data=dump_arrays(bad),headers={'Content-Type':CONTENT_TYPE},method='POST')
        with pytest.raises(error.HTTPError) as caught:
            request.urlopen(req,timeout=3)
        assert caught.value.code==400 and len(packets)==3
        before=list(_server_image_resources(server).entries)
        engine.submit=lambda _packet:False
        dropped=dict(flat,encoded_ns=np.int64(999))
        payload=dump_arrays(dropped)
        req=request.Request(f'http://127.0.0.1:{server.server_port}/v1/observations',
            data=payload,headers={'Content-Type':CONTENT_TYPE},method='POST')
        with pytest.raises(error.HTTPError) as caught:
            request.urlopen(req,timeout=3)
        assert caught.value.code==400
        assert list(_server_image_resources(server).entries)==before
        assert not (tmp_path/'vla/inputs'/(hashlib.sha256(payload).hexdigest()+'.awb')).exists()
        bad=dict(flat,model_images_blob_0=np.asarray([1,2,3],dtype=np.uint8))
        req=request.Request(f'http://127.0.0.1:{server.server_port}/v1/observations',
            data=dump_arrays(bad),headers={'Content-Type':CONTENT_TYPE},method='POST')
        with pytest.raises(error.HTTPError) as caught:
            request.urlopen(req,timeout=3)
        assert caught.value.code==400 and len(packets)==3
    finally:
        server.shutdown();server.server_close();thread.join()


@pytest.mark.parametrize('cached',[False,True])
def test_deferred_bad_current_image_is_rejected_before_network(cached):
    _legacy,flat,_history=inputs()
    flat['model_images_current_indices']=np.asarray([2,2],dtype=np.int64)
    flat['model_images_blob_2']=np.asarray([1,2,3],dtype=np.uint8)
    flat['model_images_blob_count']=np.int64(3)
    packet=decode_observation_packet(dump_arrays(flat),decode_current=False)
    engine=AsyncRTCPolicy.__new__(AsyncRTCPolicy)
    engine.validate_input=lambda _packet:None
    engine.base_camera_key,engine.wrist_camera_key='base','wrist'
    engine.mem_image_num_frames=3
    engine.mem_enabled=True
    engine.action_history_enabled=False
    engine._image_inputs=(ImageTensorCache(('base','wrist'),3,(16,16),torch.device('cpu'),
        lambda images:_decode_jpeg_window(images,writable=False),resize_with_pad_torch) if cached else None)
    with pytest.raises(ValueError,match='JPEG'):
        engine._make_batch(packet)
