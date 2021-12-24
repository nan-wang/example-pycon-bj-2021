""" Implementation of filters for images and texts"""

import numpy as np
from jina import Executor, DocumentArray, requests


class ImageReader(Executor):
    @requests
    def index_read(self, docs: 'DocumentArray', **kwargs):
        result = DocumentArray()
        for doc in docs.traverse_flat(
                traversal_paths='r',
                filter_fn=lambda d: d.mime_type in ('image/jpeg', 'image/png')):
            if doc.buffer:
                doc.convert_buffer_to_image_blob()
            elif doc.uri:
                doc.load_uri_to_image_blob()
            doc.blob = np.array(doc.blob).astype(np.uint8)
            result.append(doc)
        return result


class TextFilter(Executor):
    @requests
    def filter_text(self, docs: 'DocumentArray', **kwargs):
        return docs.traverse_flat(
            traversal_paths='r',
            filter_fn=lambda doc: doc.mime_type == 'text/plain')


class RemoveEmbedding(Executor):
    @requests
    def remove(self, docs, **kwargs):
        for doc in docs.traverse_flat(traversal_paths='r,m'):
            doc.pop('embedding', 'blob', 'buffer')
