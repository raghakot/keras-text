import shutil

from md_autogen import MarkdownAPIGenerator
from md_autogen import to_md_file

from keras_text.models import token_model
from keras_text.models import sentence_model
from keras_text.models import sequence_encoders
from keras_text.models import layers

from keras_text import data
from keras_text import generators
from keras_text import processing
from keras_text import sampling
from keras_text import utils


def generate_api_docs():
    modules = [
        token_model,
        sentence_model,
        sequence_encoders,
        layers,
        data,
        generators,
        processing,
        sampling,
        utils
    ]

    md_gen = MarkdownAPIGenerator("keras_text", "https://github.com/raghakot/keras-text/tree/master")
    for m in modules:
        md_string = md_gen.module2md(m)
        to_md_file(md_string, m.__name__, "sources")


def update_index_md():
    shutil.copyfile('../README.md', 'sources/index.md')


def copy_templates():
    shutil.rmtree('sources', ignore_errors=True)
    shutil.copytree('templates', 'sources')


if __name__ == "__main__":
    copy_templates()
    update_index_md()
    generate_api_docs()
