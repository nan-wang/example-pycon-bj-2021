import os

import streamlit as st
import requests
import numpy as np
import cv2
from math import ceil
import tempfile

DEFAULT_ENDPOINT = 'http://127.0.0.1:12345/search'

APP_NAME = 'Image-to-image Search'
APP_DESCRIPTION = 'Looking For Look like Images'

TOP_K = 2

# Query For Data
def request_and_display(query_input, top_k):
    headers = {
        'Content-Type': 'application/json',
    }

    doc = {}
    if query_input.endswith('.jpg') and os.path.exists(query_input):
        doc['uri'] = query_input
        doc['mime_type'] = 'image/jpeg'

    body = {
        'data': [doc],
        'parameters': {
            'limits': top_k
        }
    }
    response = requests.post(DEFAULT_ENDPOINT, headers=headers, json=body)
    content = response.json()
    matches = content['data']['docs'][0]['matches']
    display_photos(matches)


# Display match photos:
def display_photos(matches):
    st.markdown(
        f"""
        ### Best matches for your query:
        """
    )
    cnt = len(matches)
    row = int(ceil(cnt / 2))
    for i in range(row):
        for idx, col in enumerate(st.columns(2)):
            with col:
                match_idx = i * 2 + idx
                if match_idx >= cnt:
                    break
                uri = matches[match_idx]['uri']
                score = matches[match_idx]["scores"]['cosine']['value']
                st.markdown(f"#### No.{match_idx + 1}, Score: {score:.3f}\n ")
                st.image(uri)


def main():
    st.markdown(
        f"""
        # <a href="https://github.com/jina-ai/jina/">Jina</a>\'s {APP_NAME}
        ### {APP_DESCRIPTION}
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header('Looking For Images From Captions and vice versa')
    settings = st.sidebar.expander(label='Settings', expanded=False)
    with settings:
        endpoint = st.text_input(label='Endpoint', value=DEFAULT_ENDPOINT)
        top_k = st.number_input(label='Top K', value=TOP_K, step=1)
    st.sidebar.button('select')

    st.sidebar.markdown(
        f"""
        **This is a {APP_NAME} using the [Jina neural search framework](https://github.com/jina-ai/jina/).**
        You can search for photos using description or getting your own photo's description!

        <a href="https://github.com/jina-ai/jina/"><img src="https://github.com/alexcg1/jina-app-store-example/blob/a8f64332c6a5b3ae42df07d4bd615ff1b7ece4d9/frontend/powered_by_jina.png?raw=true" width=256></a>
        """,
        unsafe_allow_html=True,
    )

    query_image_file = None
    # Content
    query_image = st.file_uploader(
        label='select your JPEG image',
        type="jpg"
    )
    if query_image is not None:
        query_image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')

        # change the file into openCV mode
        file_bytes = np.asarray(bytearray(query_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # display
        st.image(opencv_image, channels="BGR")
        # store in certain directory
        cv2.imwrite(query_image_file.name, opencv_image)

    clicked = st.button("Search")
    if clicked:
        if not query_image_file:
            st.markdown('Please enter a query')
        else:
            request_and_display(query_image_file.name, top_k)
            if query_image_file:
                query_image_file.close()


if __name__ == '__main__':
    main()
