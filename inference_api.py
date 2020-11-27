import base64
import torch
import json
import numpy as np
import time


def ndarray_decode_base64(s, dtype="float32"):
    return np.frombuffer(bytes_decode_base64(s), dtype=dtype)


def bytes_decode_base64(s):
    if isinstance(s, str):
        s = s.encode("ascii")
    return base64.b64decode(s)


def ndarray_encode_base64(ar, astype="float32"):
    return bytes_encode_base64(ar.astype(astype).tobytes())


def bytes_encode_base64(b):
    return base64.b64encode(b).decode("ascii")


def fixlen(a, l):
    return np.pad(a, (0, l - len(a)))


def chunked(iterable, chunksize):
    chunk = []
    for elem in iterable:
        chunk.append(elem)
        if len(chunk) == chunksize:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def make_app(make_predict_func, chunksize):
    predict_func = None

    def app(environ, start_response, bench=None):
        nonlocal predict_func
        if predict_func is None:
            predict_func = make_predict_func()

        if bench is None:
            inputs = json.loads(Request(environ).get_data())
            c = chunksize
        else:
            inputs, c = bench

        s = time.time()
        outputs = []
        for batch in chunked(inputs, c):
            if bench is None:
                batch = list(map(ndarray_decode_base64, batch))
            lens = list(map(len, batch))
            batch = np.vstack([fixlen(inp, max(lens)) for inp in batch])
            try:
                with torch.no_grad():
                    pred = [
                        (ndarray_encode_base64(r[:l]) if bench is None else r[:l])
                        for r, l in zip(predict_func(torch.from_numpy(batch)).cpu().numpy(), lens)
                    ]
            except Exception as e:
                print(f"Error during denoising for inputs {len(outputs)} to {len(outputs) + len(batch)}: {e}")
                pred = [None] * len(batch)
            outputs.extend(pred)
        print("Inference of", len(inputs), "items", "took", time.time() - s)

        if bench is None:
            return Response(json.dumps(outputs))(environ, start_response)

    return app


def benchmark(app):
    data = [np.random.uniform(size=(5*16000,)).astype("float32") for _ in range(1)]
    while 1:
        app(None, None, bench=(data, 1))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="path to model")
    parser.add_argument("--host", default="127.0.0.1", type=str, help="host to serve")
    parser.add_argument("--port", required=True, type=int, help="port to serve")
    parser.add_argument("--processes", default=1, type=int, help="number of processes to spawn")
    parser.add_argument("--chunksize", default=5, type=int, help="max number of chunks to process at once")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    # MUST load model after Werkzeug forks processes, otherwise PyTorch will deadlock
    def make_predict_func():
        from models import DeepConvolutionalUNet
        net = DeepConvolutionalUNet(hidden_size=512 // 2 + 1)
        net = torch.nn.DataParallel(net)
        checkpoint = torch.load(args.model, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        return net

    app = make_app(make_predict_func, args.chunksize)

    if args.benchmark:
        benchmark(app)
    else:
        import bjoern
        from werkzeug.wrappers import Request, Response

        if args.processes > 1:
            import multiprocessing
            import functools
            procs = [multiprocessing.Process(target=functools.partial(bjoern.run, app, args.host, args.port, reuse_port=True))
                     for i in range(args.processes)]
            for p in procs:
                p.start()
            print("Ready")
            for p in procs:
                p.join()
        else:
            bjoern.run(app, args.host, args.port)
