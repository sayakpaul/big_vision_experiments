## Useful variables

```
export NAME="big_vision_exp"
export ZONE="us-central1-a"
export REGION="us-central1"
export GS_BUCKET_NAME="big_vision_exp"
```


## Set compute zone

`gcloud config set compute/zone $ZONE`

## Create a GCS bucket

`gsutil mb -l $REGION gs://$GS_BUCKET_NAME`

## Create a TPU VM (v3-8)

The needs the Cloud TPU API to be enabled beforehand.

```
gcloud alpha compute tpus tpu-vm create $NAME \
    --zone $ZONE \
    --accelerator-type v3-8 \
    --version tpu-vm-tf-2.8.0
```

## Install `big_vision`

```
git clone --branch=main https://github.com/google-research/big_vision
gcloud alpha compute tpus tpu-vm scp --recurse big_vision/big_vision $NAME: --worker=all --zone=$ZONE
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "bash big_vision/run_tpu.sh"
```

## Get the `imagenet2012` data prepared 

See here before: https://www.tensorflow.org/datasets/catalog/imagenet2012

```
import tensorflow_datasets as tfds

data_dir = "gs://imagenet-1k/tensorflow_datasets"
builder = tfds.builder("imagenet2012", data_dir=data_dir)
builder.download_and_prepare()
```

**Notice** that my GCS buxket in `data_dir` is different from what's specified in `GS_BUCKET_NAME`. This is intentional as I find it useful to segregate data files from experimental files.

## Get the `imagenet2012_real` data prepared

See here before: https://www.tensorflow.org/datasets/catalog/imagenet2012_real

```
import tensorflow_datasets as tfds

data_dir = "gs://imagenet-1k/tensorflow_datasets"
builder = tfds.builder("imagenet2012_real", data_dir=data_dir)
builder.download_and_prepare()
```

## Get the `imagenet_v2` data prepared

First install the nightly build of TensorFlow Datasets otherwise the checksums won't be matched:

`pip install -q tfds-nightly`

Then do:

```
ds = tfds.load("imagenet_v2")
```

And then move the dataset:

```
gsutil -m cp -r ~/tensorflow_datasets/imagenet_v2 gs://imagenet-1k/tensorflow_datasets/
```

Notes:

* You're running `tfds.load()` locally and your download config of `tfds` is the default one. 
* Change the target GCS bucket location as needed.


## Start training

```
gcloud alpha compute tpus tpu-vm ssh $NAME \
    --zone=$ZONE --worker=all \
    --command "TFDS_DATA_DIR=gs://imagenet-1k/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/vit_s16_i1k.py  --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'`"
```

## References

https://github.com/google-research/big_vision
