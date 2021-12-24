from jina import DocumentArray, Executor, Flow, requests


class PreprocImg(Executor):
    @requests
    def foo(self, docs, **kwargs):
        for d in docs:
            (d.load_uri_to_image_blob()  # load
             .set_image_blob_shape(shape=(375, 500))
             .set_image_blob_normalization()  # normalize color
             .set_image_blob_channel_axis(-1, 0))  # switch color axis


class EmbedImg(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import torchvision
        self.model = torchvision.models.resnet50(pretrained=True)

    @requests
    def foo(self, docs, **kwargs):
        docs.embed(self.model)


class MatchImg(Executor):
    _da = DocumentArray()

    @requests(on='/index')
    def index(self, docs, **kwargs):
        self._da.extend(docs)
        docs.clear()  # clear content to save bandwidth

    @requests(on='/search')
    def foo(self, docs, **kwargs):
        docs.match(self._da)
        for d in docs.traverse_flat('r,m'):  # only require for visualization
            d.convert_uri_to_datauri()  # convert to datauri
            d.pop('embedding', 'blob')  # remove unnecessary fields for save bandwidth


def main():
    f = Flow(port_expose=12345, protocol='http').add(uses=PreprocImg).add(uses=EmbedImg).add(uses=MatchImg)
    with f:
        f.post('/index', DocumentArray.from_files('toy_data/left/*.jpg'), show_progress=True, request_size=8)
        f.block()

if __name__ == '__main__':
    main()