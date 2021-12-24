import torchvision
from copy import deepcopy

from jina import DocumentArray
import finetuner as ft
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner


def hit_rate(da, topk=1):
    hit = 0
    for d in da:
        for m in d.matches[:topk]:
            if d.uri.split('/')[-1] == m.uri.split('/')[-1]:
                hit += 1
    return hit / len(da)


def preprocess(doc):
    return doc.load_uri_to_image_blob().set_image_blob_normalization().set_image_blob_channel_axis(-1, 0)


def assign_label_and_preprocess(doc):
    doc.tags['finetuner_label'] = doc.uri.split('/')[-1]
    return preprocess(doc)


def main():
    left_da = DocumentArray.from_files('left/*.jpg')
    right_da = DocumentArray.from_files('right/*.jpg')
    left_da.sort(key=lambda x: x.uri)
    right_da.sort(key=lambda x: x.uri)
    train_da = left_da[:500] + right_da[:500]
    train_da.apply(assign_label_and_preprocess)

    resnet = torchvision.models.resnet50(pretrained=True)

    tuned_model = ft.fit(
        model=resnet,
        train_data=train_da,
        epochs=2,
        batch_size=128,
        loss=TripletLoss(miner=TripletEasyHardMiner(neg_strategy='hard'), margin=0.3),
        learning_rate=1e-5,
        to_embedding_model=True,
        input_size=(3, 224, 224),
        num_items_per_class=2,
        layer_name='adaptiveavgpool2d_173',
        freeze=['conv2d_1', 'batchnorm2d_2', 'conv2d_5', 'batchnorm2d_6', 'conv2d_8', 'batchnorm2d_9', 'conv2d_11', 'batchnorm2d_12'],
    )

    left_da.apply(preprocess)
    right_da.apply(preprocess)

    left_da_pretrained = deepcopy(left_da)
    right_da_pretrained = deepcopy(right_da)
    # use finetuned model to create embeddings， only test data
    left_da.embed(tuned_model)
    right_da.embed(tuned_model)
    left_da.match(right_da, limit=5)
    for k in range(1, 5):
        print(f'hit@{k}:  finetuned: {hit_rate(left_da, k):.3f}')
    ft.tuner.save('finetuned.model')

    # pretrained_model = ft.tailor.to_embedding_model(
    #     model=resnet,
    #     layer_name='adaptiveavgpool2d_173',
    #     input_size=(3, 224, 224),
    # )
    # left_da_pretrained.embed(pretrained_model)
    # right_da_pretrained.embed(pretrained_model)
    # left_da_pretrained.match(right_da_pretrained, limit=5)
    #
    # print(f'pretrained emb: {left_da_pretrained[0].embedding}')
    print(f'tuned emb: {left_da[0].embedding}')
    # for k in range(1, 5):
    #     print(f'hit@{k}:  pretrained: {hit_rate(left_da_pretrained, k):.3f}')


if __name__ == '__main__':
    main()
