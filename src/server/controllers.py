from grpc import server

from src.server.server_store import server_store
from src.train.test import test_loop


def predict():
    config = server_store.config
    args = server_store.args
    results = test_loop(
        args.device,
        server_store.model,
        server_store.datamodule,
        server_store.data_loader,
    )
    return results
