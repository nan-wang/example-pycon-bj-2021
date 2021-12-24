__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import sys

import click
from jina import Flow, DocumentArray
import logging

MAX_DOCS = int(os.environ.get("JINA_MAX_DOCS", 10000))


def config():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ.setdefault('JINA_WORKSPACE', os.path.join(cur_dir, 'workspace'))
    os.environ.setdefault('JINA_LOG_LEVEL', 'INFO')
    os.environ.setdefault('JINA_PORT', str(45678))


def index(num_docs):
    flow = Flow().load_config('flows/flow-index.yml')
    with flow:
        flow.post(on='/index',
                  inputs=DocumentArray.from_files('toy_data/*.jpg')[:num_docs],
                  request_size=16,
                  show_progress=True)


def query():
    flow = Flow().load_config('flows/flow-query.yml')
    with flow:
        flow.block()


@click.command()
@click.option('--task', '-t', type=click.Choice(['index', 'index_restful', 'query_restful', 'query']), default='index')
@click.option('--num_docs', '-n', default=MAX_DOCS)
def main(task, num_docs):
    config()
    workspace = os.environ['JINA_WORKSPACE']
    logger = logging.getLogger('cross-modal-search')
    if 'index' in task:
        if os.path.exists(workspace):
            logger.error(
                f'\n +------------------------------------------------------------------------------------+ \
                    \n |                                   🤖🤖🤖                                           | \
                    \n | The directory {workspace} already exists. Please remove it before indexing again.  | \
                    \n |                                   🤖🤖🤖                                           | \
                    \n +------------------------------------------------------------------------------------+'
            )
            sys.exit(1)
    if 'query' in task:
        if not os.path.exists(workspace):
            logger.error(f'The directory {workspace} does not exist. Please index first via `python app.py -t index`')
            sys.exit(1)

    if task == 'index':
        index(num_docs)
    elif task == 'query':
        query()


if __name__ == '__main__':
    main()
